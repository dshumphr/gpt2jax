import jax
import jax.numpy as jnp
import tiktoken

def check_nan(tensor, name):
    if jnp.isnan(tensor).any():
        print(f"NaN detected in {name}")

class Transformer:
    def __init__(self, v, h, e, n, l):
        self.toke_ve = jax.random.normal(jax.random.PRNGKey(0), (v, e)) * 0.02
        self.pose_ve = jax.random.normal(jax.random.PRNGKey(1), (l, e)) * 0.02
        self.blocks = [Block(h, e, n) for _ in range(n)]
        self.lf = LayerNorm(e)

    def __call__(self, x_blv):
        tokemb_ble = jnp.einsum('blv,ve->ble', x_blv, self.toke_ve)
        check_nan(tokemb_ble, "tokemb_ble")
        posemb_ble = jnp.einsum('bl,le->ble', jnp.arange(x_blv.shape[1])[None,:], self.pose_ve[:x_blv.shape[1]])
        check_nan(posemb_ble, "posemb_ble")
        emb_ble = tokemb_ble + posemb_ble
        check_nan(emb_ble, "emb_ble")
        o_ble = emb_ble
        for i, block in enumerate(self.blocks):
            o_ble = block(o_ble)
            check_nan(o_ble, f"Block {i} output")
        o_ble = self.lf(o_ble)
        check_nan(o_ble, "Final layer norm output")
        logits_blv = jnp.einsum('ble,ve->blv', o_ble, self.toke_ve)
        check_nan(logits_blv, "Final logits")
        return logits_blv

class MHA:
    def __init__(self, h, e):
        key = jax.random.PRNGKey(2)
        keys = jax.random.split(key, 4)
        self.wq_ehk = jax.random.normal(keys[0], (e, h, e//h)) * 0.02
        self.wk_ehk = jax.random.normal(keys[1], (e, h, e//h)) * 0.02
        self.wv_ehk = jax.random.normal(keys[2], (e, h, e//h)) * 0.02
        self.wo_hke = jax.random.normal(keys[3], (h, e//h, e)) * 0.02
        self.b_e = jnp.zeros(e)

    def __call__(self, x_ble):
        q_blhk = jnp.einsum('ble,ehk->blhk', x_ble, self.wq_ehk)
        k_blhk = jnp.einsum('ble,ehk->blhk', x_ble, self.wk_ehk)
        v_blhk = jnp.einsum('ble,ehk->blhk', x_ble, self.wv_ehk)
        check_nan(q_blhk, "q_blhk")
        check_nan(k_blhk, "k_blhk")
        check_nan(v_blhk, "v_blhk")
        attn = jnp.einsum('blhk,bmhk->bhlm', q_blhk, k_blhk)
        b, l, h, k = q_blhk.shape
        attn = attn / jnp.sqrt(k)
        check_nan(attn, "attn pre-mask")
        mask = jnp.triu(jnp.ones((l, l)), k=1).astype(x_ble.dtype)
        attn = jnp.where(mask[None, None, :, :], -jnp.inf, attn)
        check_nan(attn, "attn post-mask")
        attn = jax.nn.softmax(attn, axis=-1)
        check_nan(attn, "attn post-softmax")
        values = jnp.einsum('blhk,bhlm->blhk', v_blhk, attn)
        out_ble = jnp.einsum('blhk,hke->ble', values, self.wo_hke) + self.b_e
        check_nan(out_ble, "MHA output")
        return out_ble

class Block:
    def __init__(self, h, e, layers):
        self.ffn = FFN(e, layers)
        self.ln1 = LayerNorm(e)
        self.attn = MHA(h, e)
        self.ln2 = LayerNorm(e)

    def __call__(self, x_ble):
        ln1_out = self.ln1(x_ble)
        check_nan(ln1_out, "LN1 output")
        attn_out = self.attn(ln1_out)
        check_nan(attn_out, "Attention output")
        x_ble += attn_out
        check_nan(x_ble, "Post-attention residual")
        ln2_out = self.ln2(x_ble)
        check_nan(ln2_out, "LN2 output")
        ffn_out = self.ffn(ln2_out)
        check_nan(ffn_out, "FFN output")
        x_ble += ffn_out
        check_nan(x_ble, "Post-FFN residual")
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
        check_nan(y_ble, "LayerNorm output")
        return y_ble

class FFN:
    def __init__(self, e, layers):
        self.e = e
        key = jax.random.PRNGKey(3)
        keys = jax.random.split(key, 2)
        self.fc = jax.random.normal(keys[0], (e, 4*e)) * 0.02 / jnp.sqrt(2 * layers)
        self.proj = jax.random.normal(keys[1], (4*e, e)) * 0.02 / jnp.sqrt(2 * layers)
        self.b1 = jnp.zeros(4*e)
        self.b2 = jnp.zeros(e)

    def __call__(self, x_ble):
        x_blk = jnp.einsum('ble,ek->blk', x_ble, self.fc) + self.b1
        check_nan(x_blk, "FFN intermediate")
        x_blk = jax.nn.gelu(x_blk)
        check_nan(x_blk, "FFN post-GELU")
        y_ble = jnp.einsum('blk,ke->ble', x_blk, self.proj) + self.b2
        check_nan(y_ble, "FFN output")
        return y_ble

class Embedding:
    def __init__(self, v, e):
        self.w = jax.random.normal(jax.random.PRNGKey(4), (v, e)) * 0.02

    def __call__(self, x_bl):
        y_ble = self.w[x_bl]
        check_nan(y_ble, "Embedding output")
        return y_ble

def print_shape(model):
    print("Transformer shapes:")
    print(f"toke_ve: {model.toke_ve.shape}")
    print(f"pose_ve: {model.pose_ve.shape}")
    print(f"Number of blocks: {len(model.blocks)}")

    for i, block in enumerate(model.blocks):
        print(f"\nBlock {i} shapes:")
        print(f"FFN fc: {block.ffn.fc.shape}")
        print(f"FFN proj: {block.ffn.proj.shape}")
        print(f"MHA wq: {block.attn.wq_ehk.shape}")
        print(f"MHA wk: {block.attn.wk_ehk.shape}")
        print(f"MHA wv: {block.attn.wv_ehk.shape}")
        print(f"MHA wo: {block.attn.wo_hke.shape}")

# Builtin random.choice is seg faulting, maybe m1 mac issue? Just write it ourselves.
def custom_choice(key, a, p):
    """Custom implementation of random choice"""
    cum_probs = jnp.cumsum(p)
    r = jax.random.uniform(key)
    return jnp.argmin(cum_probs < r)

def sample(model, length):
    enc = tiktoken.get_encoding("gpt2")
    x = jnp.array([enc.encode("Hello, I'm a language model,")])
    x = jax.nn.one_hot(x, vocab_size)
    for i in range(length):
        logits = model(x)
        probs = jax.nn.softmax(logits[:, -1, :], axis=-1)
        topk = jnp.argsort(probs[0])[-10:]
        topk_probs = probs[0][topk]
        topk_probs = topk_probs / jnp.sum(topk_probs)  # Renormalize
        if jnp.isnan(topk_probs).any():
            print("NaN detected in probs!")
            break
        tok = custom_choice(jax.random.PRNGKey(i), a=10, p=topk_probs)
        tok = topk[tok]
        x = jnp.concatenate([x, jax.nn.one_hot(jnp.array([[tok]]), vocab_size)], axis=1)
    return enc.decode(jnp.argmax(x[0], axis=-1).tolist())

# TODO optimizer, training loop

# Hparams
epochs = 50
heads = 12
layers = 12
hidden_size = 768
vocab_size = 50257
B = 32
L = 1024

# Init Transformer and print all shapes to ensure they align with expectations
model = Transformer(vocab_size, heads, hidden_size, layers, L)

#print_shape(model)
print("\nSample output:")
print(sample(model, 50))