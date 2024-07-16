import jax
import jax.numpy as jnp

# Hparams
epochs = 50
heads = 12
layers = 12
hidden_size = 768
vocab_size = 50257
B = 32
L = 1024

# Modules: impl 1 by 1
class Transformer:
    def __init__(self, v, h, e, n, l):
        # embed, n blocks, final output
        self.toke_ve = jnp.ones((v, e))
        self.pose_ve = jnp.ones((l, e))
        self.blocks = [Block(h, e) for _ in range(n)]
        self.lf = LayerNorm(e)
        # out is toke_ve reversed

    def __call__(self, x_blv):
        tokemb_ble = jnp.einsum('blv,ve->ble', x_blv, self.toke_ve)
        posemb_ble = jnp.einsum('bl,le->ble', jnp.arange(x_blv.shape[1])[None,:], self.pose_ve)
        emb_ble = tokemb_ble + posemb_ble
        o_ble = emb_ble
        # TODO try scan instead?
        for block in self.blocks:
            o_ble = block(o_ble)
        o_ble = jax.lax.scan()
        o_ble = self.lf(o_ble)        
        logits_blv = jnp.einsum('ble,ve->blv', o_ble, self.toke_ve)
        return logits_blv

class MHA:
    def __init__(self, h, e):
        self.wq_ehk = jnp.ones((e, h, e//h))
        self.wk_ehk = jnp.ones((e, h, e//h))
        self.wv_ehk = jnp.ones((e, h, e//h))
        self.wo_hke = jnp.ones((h, e//h, e))

    def __call__(self, x_ble):
        q_blhk = jnp.einsum('ble,ehk->blhk', x_ble, self.wq_ehk)
        k_blhk = jnp.einsum('ble,ehk->blhk', x_ble, self.wk_ehk)
        v_blhk = jnp.einsum('ble,ehk->blhk', x_ble, self.wv_ehk)
        # cross qk, then scale
        attn = jnp.einsum('blhk,bmhk->bhlm', q_blhk, k_blhk)
        b, l, h, k = q_blhk.shape
        attn = attn / jnp.sqrt(k)
        # mask n softmax
        mask = jnp.triu(jnp.ones((l, l)), k=1).astype(x_ble.dtype)
        attn = attn - jnp.inf * mask[None, None, :, :]
        attn = jax.nn.softmax(attn, axis=-1)
        # cross w values n project
        values = jnp.einsum('blhk,bhlm->blhk', v_blhk, attn)
        out_ble = jnp.einsum('blhk,hke->ble', values, self.wo_hke)
        return out_ble

class Block:
    def __init__(self, h, e):
        self.ffn = FFN(e)
        self.ln1 = LayerNorm(e)
        self.attn = MHA(h, e)
        self.ln2 = LayerNorm(e)

    def __call__(self, x_ble):
        x_ble += self.attn(self.ln1(x_ble))
        x_ble += self.ffn(self.ln2(x_ble))
        return x_ble


class LayerNorm:
    def __init__(self, e, eps=1e-5):
        self.eps = eps
        self.gamma_e = jnp.ones(e)
        self.beta_e = jnp.zeros(e)

    def __call__(self, x_ble):
        mean_bl = jnp.mean(x_ble, axis=-1, keepdims=True)
        var_bl = jnp.var(x_ble, axis=-1, keepdims=True)
        norm_ble = (x_ble - mean_bl) / jnp.sqrt(var_bl + self.eps)
        y_ble = self.gamma_e * norm_ble + self.beta_e
        return y_ble

class FFN:
    def __init__(self, e):
        self.e = e
        self.fc = jnp.ones((e, 4*e))
        self.proj = jnp.ones((4*e, e))

    def __call__(self, x_ble):
        x_blk = jnp.einsum('ble,ek->blk', x_ble, self.fc)
        x_blk = jax.nn.gelu(x_blk)
        y_ble = jnp.einsum('blk,ke->ble', x_blk, self.proj)
        return y_ble

class Embedding:
    def __init__(self, v, e):
        self.w = jnp.ones((v, e))

    def __call__(self, x_bl):
        y_ble = self.w[x_bl]
        return y_ble

# TODO Inits, optimizer, training loop, sample loop

print(jax.numpy.arange(10))