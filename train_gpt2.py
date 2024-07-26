import jax
import jax.numpy as jnp
import tiktoken
import optax
from typing import Dict, Any
import time
import math
from flash_attention_jax import causal_flash_attention
import os

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
            'b_e': jnp.zeros(e)
        }

    @staticmethod
    def apply(params, x_ble):
        q_blhk = jnp.einsum('ble,ehk->blhk', x_ble, params['wq_ehk'])
        k_blhk = jnp.einsum('ble,ehk->blhk', x_ble, params['wk_ehk'])
        v_blhk = jnp.einsum('ble,ehk->blhk', x_ble, params['wv_ehk'])
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
        q_bhlk = jnp.einsum('ble,ehk->bhlk', x_ble, params['wq_ehk'])
        k_bhlk = jnp.einsum('ble,ehk->bhlk', x_ble, params['wk_ehk'])
        v_bhlk = jnp.einsum('ble,ehk->bhlk', x_ble, params['wv_ehk'])
        values = causal_flash_attention(q_bhlk, k_bhlk, v_bhlk)
        out_ble = jnp.einsum('bhlk,hke->ble', values, params['wo_hke']) + params['b_e']
        """
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

@jax.jit
def compute_loss_and_grads(params, batch):
    def loss_fn(params):
        logits = Transformer.apply(params, jax.nn.one_hot(batch[:, :-1], vocab_size))
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch[:, 1:])
        return jnp.mean(loss)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

@jax.jit
def update_params(params, grads, optimizer_state):
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state

def custom_choice(key, a, p):
    """Custom implementation of random choice"""
    cum_probs = jnp.cumsum(p)
    r = jax.random.uniform(key)
    return jnp.argmin(cum_probs < r)

def sample(model, length):
    enc = tiktoken.get_encoding("gpt2")
    x = jnp.array([enc.encode("All:")])
    x = jax.nn.one_hot(x, vocab_size)
    for i in range(length):
        logits = Transformer.apply(params, x)
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

def load_tokens(filename):
    npt = jnp.load(filename)
    return npt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).reshape(B, T)
        y = (buf[1:]).reshape(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

# Hparams
heads = 12
layers = 12
hidden_size = 768
vocab_size = 50304
B = 4
L = 1024
max_steps = 19073

# Initialize model parameters
key = jax.random.PRNGKey(0)
params = Transformer.init(key, vocab_size, heads, hidden_size, layers, L)

# Learning rate scheduler using optax
warmup_steps = 715
max_lr = 18e-4
min_lr = max_lr * 0.1
schedule_fn = optax.warmup_cosine_decay_schedule(
    init_value=min_lr,
    peak_value=max_lr,
    warmup_steps=warmup_steps,
    decay_steps=max_steps,
    end_value=min_lr
)

# Training loop with gradient accumulation
step = 0
total_tokens = 524288
tokens_per_batch = B * L
accumulation_steps = total_tokens // tokens_per_batch
accumulated_grads = None
accumulated_loss = 0.0

# Initialize optimizer
weight_decay = 0.1
mask = jax.tree_map(lambda x: x.ndim >= 2, params)
optimizer = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule_fn, b1=0.9, b2=0.95, weight_decay=weight_decay, mask=mask)
    ),
    every_k_schedule=accumulation_steps,
)
optimizer_state = optimizer.init(params)

start_time = time.time()
dataloader = DataLoaderLite(B, L, 'train')
valloader = DataLoaderLite(B, L, 'val')
with open("loss_history.txt", "w") as loss_file:
    for step in range(max_steps):
        batch, _ = dataloader.next_batch()
        loss, grads = compute_loss_and_grads(params, batch)
        accumulated_loss += loss
        params, optimizer_state = update_params(params, grads, optimizer_state)

        if (step + 1) % accumulation_steps == 0:
            # Calculate tokens/s
            tokens_processed = ((step + 1) * tokens_per_batch)
            elapsed_time = time.time() - start_time
            tokens_per_second = tokens_processed / elapsed_time

            print(f"Step {step + 1}, Train Loss: {accumulated_loss / accumulation_steps:.4f}, Tokens/s: {tokens_per_second:.2f}")
            loss_file.write(f"Step {step + 1}, Train Loss: {accumulated_loss / accumulation_steps:.4f}, Tokens/s: {tokens_per_second:.2f}\n")
            accumulated_loss = 0.0

        if (step + 1) % 250 == 0:
            val_loss = 0.0
            for _ in range(accumulation_steps):
                val_batch, _ = valloader.next_batch()
                val_loss_step, _ = compute_loss_and_grads(params, val_batch)
                val_loss += val_loss_step
            val_loss /= accumulation_steps
            print(f"Step {step + 1}, Validation Loss: {val_loss:.4f}")
            loss_file.write(f"Step {step + 1}, Validation Loss: {val_loss:.4f}\n")
            loss_file.flush()

print("\nTraining completed. Final sample output:")
print(sample(params, 10))