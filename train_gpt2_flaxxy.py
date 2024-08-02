import jax
import jax.numpy as jnp
import tiktoken
import optax
from typing import Dict, Any
import time
from flash_attention_jax import causal_flash_attention
from flax import linen as nn
from flax.training import train_state

# Flax version for speed/correctness test
# TODO fix up initializers, etc to match regular version

def check_nan(tensor, name):
    if jnp.isnan(tensor).any():
        print(f"NaN detected in {name}")

class MHA(nn.Module):
    num_heads: int
    
    @nn.compact
    def __call__(self, x):
        b, l, e = x.shape
        h = self.num_heads
        k = e // h
        
        wq = self.param('wq', nn.initializers.normal(0.02), (e, h, k))
        wk = self.param('wk', nn.initializers.normal(0.02), (e, h, k))
        wv = self.param('wv', nn.initializers.normal(0.02), (e, h, k))
        wo = self.param('wo', nn.initializers.normal(0.02), (h, k, e))
        b = self.param('b', nn.initializers.zeros, (e,))
        
        q = jnp.einsum('ble,ehk->bhlk', x, wq)
        k = jnp.einsum('ble,ehk->bhlk', x, wk)
        v = jnp.einsum('ble,ehk->bhlk', x, wv)
        
        values = causal_flash_attention(q, k, v)
        
        out = jnp.einsum('bhlk,hke->ble', values, wo) + b
        return out

class FFN(nn.Module):
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        h = self.hidden_dim
        x = nn.Dense(features=4*h, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.gelu(x)
        x = nn.Dense(features=h, kernel_init=nn.initializers.normal(0.02))(x)
        return x

class Block(nn.Module):
    num_heads: int
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = x + MHA(num_heads=self.num_heads)(nn.LayerNorm()(x))
        x = x + FFN(hidden_dim=self.hidden_dim)(nn.LayerNorm()(x))
        return x

class Transformer(nn.Module):
    vocab_size: int
    num_heads: int
    num_layers: int
    hidden_dim: int
    max_len: int
    
    @nn.compact
    def __call__(self, x):
        b, l = x.shape
        x = nn.Embed(self.vocab_size, self.hidden_dim)(x)
        x = x + self.param('pos_emb', nn.initializers.normal(0.02), (self.max_len, self.hidden_dim))[:l]
        
        for _ in range(self.num_layers):
            x = Block(num_heads=self.num_heads, hidden_dim=self.hidden_dim)(x)
        
        x = nn.LayerNorm()(x)
        logits = nn.Dense(features=self.vocab_size, use_bias=False, kernel_init=nn.initializers.normal(0.02))(x)
        return logits

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn(params, batch[:, :-1])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch[:, 1:])
        return jnp.mean(loss)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def custom_choice(key, a, p):
    cum_probs = jnp.cumsum(p)
    r = jax.random.uniform(key)
    return jnp.argmin(cum_probs < r)

def sample(state, length):
    enc = tiktoken.get_encoding("gpt2")
    x = jnp.array([enc.encode("All:")])
    for i in range(length):
        logits = state.apply_fn(state.params, x)
        probs = jax.nn.softmax(logits[:, -1, :], axis=-1)
        topk = jnp.argsort(probs[0])[-10:]
        topk_probs = probs[0][topk]
        topk_probs = topk_probs / jnp.sum(topk_probs)
        if jnp.isnan(topk_probs).any():
            print("NaN detected in probs!")
            break
        tok = custom_choice(jax.random.PRNGKey(i), a=10, p=topk_probs)
        tok = topk[tok]
        x = jnp.concatenate([x, jnp.array([[tok]])], axis=1)
    return enc.decode(x[0].tolist())

def get_batches(B, L):
    with open('input.txt', 'r') as f:
        text = f.read()
    
    enc = tiktoken.get_encoding("gpt2")
    data = jnp.array(enc.encode(text))
    
    n = len(data)
    batch_size = B * L
    n_batches = n // batch_size
    
    for i in range(n_batches):
        batch = data[i*batch_size : (i+1)*batch_size]
        batch = batch.reshape(B, L)
        yield batch

# Hparams
epochs = 5
heads = 12
layers = 12
hidden_size = 768
vocab_size = 50304
B = 4
L = 1024

# Initialize model and optimizer
model = Transformer(vocab_size=vocab_size, num_heads=heads, num_layers=layers, hidden_dim=hidden_size, max_len=L)
learning_rate = 3e-4
optimizer = optax.chain(
   optax.clip_by_global_norm(1.0),
   optax.adamw(learning_rate, b1=0.9, b2=0.95),
)

key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, L), dtype=jnp.int32))
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# Training loop
for epoch in range(epochs):
    start_time = time.time()
    for batch in get_batches(B, L):
        loss, state = train_step(state, batch)
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}, Duration: {epoch_duration:.2f} seconds")

print("\nTraining completed. Final sample output:")
print(sample(state, 10))