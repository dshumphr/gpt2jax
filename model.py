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
    def init(key, v, h, e, n, l):
        key, subkey = jax.random.split(key)
        toke_ve = jax.random.normal(subkey, (v, e)) * 0.02
        
        key, subkey = jax.random.split(key)
        pose_ve = jax.random.normal(subkey, (l, e)) * 0.02
        
        blocks = [Block.init(key, h, e, n) for _ in range(n)]
        lf = LayerNorm.init(key, e)
        
        return {
            'toke_ve': toke_ve,
            'pose_ve': pose_ve,
            'blocks': blocks,
            'lf': lf
        }

    @staticmethod
    def apply(params, x_blv):
        tokemb_ble = jnp.einsum('blv,ve->ble', x_blv, params['toke_ve'])
        posemb_ble = jnp.einsum('bl,le->ble', jnp.arange(x_blv.shape[1])[None,:], params['pose_ve'][:x_blv.shape[1]])
        emb_ble = tokemb_ble + posemb_ble
        o_ble = emb_ble
        for i, block_params in enumerate(params['blocks']):
            o_ble = Block.apply(block_params, o_ble)
        o_ble = LayerNorm.apply(params['lf'], o_ble)
        logits_blv = jnp.einsum('ble,ve->blv', o_ble, params['toke_ve'])
        return logits_blv

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
    def apply(params, x_ble):
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
        values = causal_flash_attention(q_bhlk, k_bhlk, v_bhlk)
        out_ble = jnp.einsum('bhlk,hke->ble', values, params['wo_hke']) + params['b_e']
        return out_ble

class Block:
    @staticmethod
    def init(key, h, e, layers):
        key, *subkeys = jax.random.split(key, 5)
        return {
            'ffn': FFN.init(subkeys[0], e, layers),
            'ln1': LayerNorm.init(subkeys[1], e),
            'attn': MHA.init(subkeys[2], h, e, layers),
            'ln2': LayerNorm.init(subkeys[3], e)
        }

    @staticmethod
    def apply(params, x_ble):
        ln1_out = LayerNorm.apply(params['ln1'], x_ble)
        attn_out = MHA.apply(params['attn'], ln1_out)
        x_ble += attn_out
        ln2_out = LayerNorm.apply(params['ln2'], x_ble)
        ffn_out = FFN.apply(params['ffn'], ln2_out)
        x_ble += ffn_out
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