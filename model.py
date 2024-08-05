import jax
import jax.numpy as jnp
from typing import Dict, Any
from flash_attention_jax import causal_flash_attention

#jax.config.update("jax_debug_nans", True)
def check_nan(tensor, name):
    if jnp.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(tensor)

class Transformer:
    @staticmethod
    def init(key, v, h, e, n, l, use_rmsnorm=False, use_swiglu=False):
        key, subkey = jax.random.split(key)
        toke_ve = jax.random.normal(subkey, (v, e)) * 0.02
        
        key, subkey = jax.random.split(key)
        pose_le = jax.random.normal(subkey, (l, e)) * 0.02
        
        blocks = [Block.init(key, h, e, n, use_rmsnorm, use_swiglu) for _ in range(n)]
        lf = RMSNorm.init(key, e) if use_rmsnorm else LayerNorm.init(key, e)
        
        return {
            'toke_ve': toke_ve,
            'pose_le': pose_le,
            'blocks': blocks,
            'lf': lf,
        }

    @staticmethod
    def apply(params, x_bl, use_rope=False):
        tokemb_ble = jnp.take(params['toke_ve'], x_bl, axis=0)
        if not use_rope:
            posemb_ble = jnp.take(params['pose_le'], jnp.arange(x_bl.shape[1]), axis=0)
            emb_ble = tokemb_ble + posemb_ble
        else:
            emb_ble = tokemb_ble
        o_ble = emb_ble
        for i, block_params in enumerate(params['blocks']):
            o_ble = Block.apply(block_params, o_ble, use_rope)
        o_ble = RMSNorm.apply(params['lf'], o_ble) if isinstance(params['lf'], dict) and 'gamma_e' in params['lf'] else LayerNorm.apply(params['lf'], o_ble)
        logits_blv = jnp.einsum('ble,ve->blv', o_ble, params['toke_ve'])
        return logits_blv

def apply_rope(x, dim, base=1000000):
    b, l, h, d = x.shape
    position = jnp.arange(l)[None, :, None, None]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * -(jnp.log(base) / dim))
    div_term = div_term[None, None, None, :]
    sin_inp = position * div_term
    emb = jnp.concatenate([jnp.sin(sin_inp), jnp.cos(sin_inp)], axis=-1)
    return x * emb

class MHA:
    @staticmethod
    def init(key, h, e, layers):
        key, *subkeys = jax.random.split(key, 5)
        return {
            'wq_ehk': jax.random.normal(subkeys[0], (e, h, e//h)) * 0.02,
            'wk_ehk': jax.random.normal(subkeys[1], (e, h, e//h)) * 0.02,
            'wv_ehk': jax.random.normal(subkeys[2], (e, h, e//h)) * 0.02,
            'wo_hke': jax.random.normal(subkeys[3], (h, e//h, e)) * 0.02 / jnp.sqrt(2 * layers),
            'b_e': jnp.zeros(e),
            'bq_hk': jnp.zeros((h, e//h)),
            'bk_hk': jnp.zeros((h, e//h)),
            'bv_hk': jnp.zeros((h, e//h))
        }

    @staticmethod
    def apply(params, x_ble, rope=False):
        """
        q_blhk = jnp.einsum('ble,ehk->blhk', x_ble, params['wq_ehk']) + params['bq_hk'][None, None, :, :]
        k_blhk = jnp.einsum('ble,ehk->blhk', x_ble, params['wk_ehk']) + params['bk_hk'][None, None, :, :]
        v_blhk = jnp.einsum('ble,ehk->blhk', x_ble, params['wv_ehk']) + params['bv_hk'][None, None, :, :]
        attn = jnp.einsum('blhk,bmhk->bhlm', q_blhk, k_blhk)
        b, l, h, k = q_blhk.shape
        attn = attn / jnp.sqrt(k)
        mask = jnp.triu(jnp.ones((l, l)), k=1).astype(x_ble.dtype)
        attn = jnp.where(mask[None, None, :, :], -jnp.inf, attn)
        attn = jax.nn.softmax(attn, axis=-1)
        values = jnp.einsum('blhk,bhlm->blhk', v_blhk, attn)
        out_ble = jnp.einsum('blhk,hke->ble', values, params['wo_hke']) + params['b_e']

        """
        #flash attn
        q_bhlk = jnp.einsum('ble,ehk->bhlk', x_ble, params['wq_ehk']) + params['bq_hk'][None, :, None, :]
        k_bhlk = jnp.einsum('ble,ehk->bhlk', x_ble, params['wk_ehk']) + params['bk_hk'][None, :, None, :]
        v_bhlk = jnp.einsum('ble,ehk->bhlk', x_ble, params['wv_ehk']) + params['bv_hk'][None, :, None, :]
        
        # Apply ROPE to q and k
        if rope:
            q_bhlk = apply_rope(q_bhlk, q_bhlk.shape[-1])
            k_bhlk = apply_rope(k_bhlk, k_bhlk.shape[-1])
        
        values = causal_flash_attention(q_bhlk, k_bhlk, v_bhlk)
        out_ble = jnp.einsum('bhlk,hke->ble', values, params['wo_hke']) + params['b_e']
        return out_ble

class Block:
    @staticmethod
    def init(key, h, e, layers, use_rmsnorm=False, use_swiglu=False):
        key, *subkeys = jax.random.split(key, 5)
        return {
            'ffn': Swiglu.init(subkeys[0], e, layers) if use_swiglu else FFN.init(subkeys[0], e, layers),
            'ln1': RMSNorm.init(subkeys[1], e) if use_rmsnorm else LayerNorm.init(subkeys[1], e),
            'attn': MHA.init(subkeys[2], h, e, layers),
            'ln2': RMSNorm.init(subkeys[3], e) if use_rmsnorm else LayerNorm.init(subkeys[3], e)
        }

    @staticmethod
    def apply(params, x_ble, use_rope=False):
        ln1_out = RMSNorm.apply(params['ln1'], x_ble) if isinstance(params['ln1'], dict) and 'gamma_e' in params['ln1'] else LayerNorm.apply(params['ln1'], x_ble)
        attn_out = MHA.apply(params['attn'], ln1_out, use_rope)
        x_ble = x_ble + attn_out
        ln2_out = RMSNorm.apply(params['ln2'], x_ble) if isinstance(params['ln2'], dict) and 'gamma_e' in params['ln2'] else LayerNorm.apply(params['ln2'], x_ble)
        ffn_out = Swiglu.apply(params['ffn'], ln2_out) if isinstance(params['ffn'], dict) and 'w1' in params['ffn'] else FFN.apply(params['ffn'], ln2_out)
        x_ble = x_ble + ffn_out
        return x_ble

class LayerNorm:
    eps = 1e-5

    @staticmethod
    def init(key, e):
        return {
            'gamma_e': jnp.ones(e),
            'beta_e': jnp.zeros(e),
        }

    @staticmethod
    def apply(params, x_ble):
        mean_bl = jnp.mean(x_ble, axis=-1, keepdims=True)
        var_bl = jnp.var(x_ble, axis=-1, keepdims=True)
        norm_ble = (x_ble - mean_bl) * jax.lax.rsqrt(var_bl + LayerNorm.eps)
        y_ble = params['gamma_e'] * norm_ble + params['beta_e']
        return y_ble

class RMSNorm:
    eps = 1e-5

    @staticmethod
    def init(key, e):
        return {
            'gamma_e': jnp.ones(e),
        }

    @staticmethod
    def apply(params, x_ble):
        ms = jnp.mean(jnp.square(x_ble), axis=-1, keepdims=True)
        norm_ble = x_ble * jax.lax.rsqrt(ms + RMSNorm.eps)
        y_ble = params['gamma_e'] * norm_ble
        return y_ble

class FFN:
    @staticmethod
    def init(key, e, layers):
        key, *subkeys = jax.random.split(key, 3)
        return {
            'fc': jax.random.normal(subkeys[0], (e, 4*e)) * 0.02,
            'proj': jax.random.normal(subkeys[1], (4*e, e)) * 0.02 / jnp.sqrt(2 * layers),
            'b1': jnp.zeros(4*e),
            'b2': jnp.zeros(e)
        }

    @staticmethod
    def apply(params, x_ble):
        x_blk = jnp.einsum('ble,ek->blk', x_ble, params['fc']) + params['b1']
        x_blk = jax.nn.gelu(x_blk)
        y_ble = jnp.einsum('blk,ke->ble', x_blk, params['proj']) + params['b2']
        return y_ble

class Swiglu:
    @staticmethod
    def init(key, e, layers):
        key, *subkeys = jax.random.split(key, 4)
        return {
            'w1': jax.random.normal(subkeys[0], (e, 4*e)) * 0.02,
            'w2': jax.random.normal(subkeys[1], (e, 4*e)) * 0.02,
            'w3': jax.random.normal(subkeys[2], (4*e, e)) * 0.02 / jnp.sqrt(2 * layers),
        }

    @staticmethod
    def apply(params, x_ble):
        x_blk1 = jax.nn.silu(jnp.einsum('ble,ek->blk', x_ble, params['w1']))
        x_blk2 = jnp.einsum('ble,ek->blk', x_ble, params['w2'])
        x_blk = x_blk1 * x_blk2
        return jnp.einsum('blk,ke->ble', x_blk, params['w3'])