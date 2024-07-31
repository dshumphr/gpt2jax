import jax
import jax.numpy as jnp
import optax
import torch
import numpy as np
import math
import time
from model import Transformer
from tmodel import GPT, GPTConfig

# Shared hyperparameters
vocab_size = 1000
n_layers = 2
n_heads = 4
d_model = 128
max_seq_len = 64
batch_size = 8

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19000
weight_decay = 0.1
accumulation_steps = 32
init_lr = max_lr / warmup_steps

# JAX implementation (keep as is)

# PyTorch implementation
# Initialize models
jax_key = jax.random.PRNGKey(42)
jax_params = Transformer.init(jax_key, vocab_size, n_heads, d_model, n_layers, max_seq_len)

config = GPTConfig(
    block_size = max_seq_len,
    vocab_size = vocab_size,
    n_layer = n_layers,
    n_head = n_heads,
    n_embd = d_model,
)
torch_model = GPT(config)

# Copy JAX weights to PyTorch model
"""
def copy_weights(jax_params, torch_model):
    torch_model.wte.weight.data = torch.tensor(jax_params['wte'])
    torch_model.wpe.weight.data = torch.tensor(jax_params['wpe'])
    torch_model.ln_f.weight.data = torch.tensor(jax_params['ln_f']['g'])
    torch_model.ln_f.bias.data = torch.tensor(jax_params['ln_f']['b'])
    
    for i, block in enumerate(torch_model.h):
        block.self_attn.in_proj_weight.data = torch.tensor(jax_params['h'][i]['attn']['c_attn'].T)
        block.self_attn.out_proj.weight.data = torch.tensor(jax_params['h'][i]['attn']['c_proj'].T)
        block.linear1.weight.data = torch.tensor(jax_params['h'][i]['mlp']['c_fc'].T)
        block.linear2.weight.data = torch.tensor(jax_params['h'][i]['mlp']['c_proj'].T)
        block.norm1.weight.data = torch.tensor(jax_params['h'][i]['ln_1']['g'])
        block.norm1.bias.data = torch.tensor(jax_params['h'][i]['ln_1']['b'])
        block.norm2.weight.data = torch.tensor(jax_params['h'][i]['ln_2']['g'])
        block.norm2.bias.data = torch.tensor(jax_params['h'][i]['ln_2']['b'])
copy_weights(jax_params, torch_model)
"""

# Optimizers and learning rate schedule
def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

schedule_fn = optax.warmup_cosine_decay_schedule(
    init_value=init_lr,
    peak_value=max_lr,
    warmup_steps=warmup_steps,
    decay_steps=max_steps,
    end_value=min_lr
)

mask = jax.tree_map(lambda x: x.ndim >= 2, jax_params)
jax_optimizer = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule_fn, b1=0.9, b2=0.95, weight_decay=weight_decay, mask=mask)
    ),
    every_k_schedule=accumulation_steps,
)
jax_opt_state = jax_optimizer.init(jax_params)

torch_optimizer = torch.optim.AdamW(torch_model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=weight_decay)

# Loss functions
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
    updates, optimizer_state = jax_optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state

# Training loop
for step in range(max_steps):
    jax_loss_accum = 0.0
    torch_loss_accum = 0.0

    for _ in range(accumulation_steps):
        # Generate random batch
        buf = np.random.randint(0, vocab_size, (batch_size*max_seq_len+1))
        x = (buf[:-1]).reshape(batch_size, max_seq_len)
        y = (buf[1:]).reshape(batch_size, max_seq_len)

        # JAX update
        jax_loss, jax_grads = compute_loss_and_grads(jax_params, x)
        jax_loss_accum += jax_loss

        # PyTorch update
        torch_x = torch.tensor(x)
        torch_y = torch.tensor(y)
        torch_logits, torch_loss = torch_model(torch_x, torch_y)
        torch_loss_accum += torch_loss.item()
        (torch_loss / accumulation_steps).backward()

        # JAX optimizer step
        jax_params, jax_opt_state = update_params(jax_params, jax_grads, jax_opt_state)

    # PyTorch optimizer step
    lr = get_lr(step)
    for param_group in torch_optimizer.param_groups:
        param_group['lr'] = lr
    torch_optimizer.step()
    torch_optimizer.zero_grad()

    # Compare results
    if step % 10 == 0:
        print(f"Step {step}:")
        print(f"  JAX loss: {jax_loss_accum / accumulation_steps:.6f}")
        print(f"  PyTorch loss: {torch_loss_accum / accumulation_steps:.6f}")
        
        # Compare parameters
        """
        for name, param in torch_model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                jax_param = jax_params['wte'] if name == 'wte.weight' else jax_params['wpe'] if name == 'wpe.weight' else None
                if jax_param is None:
                    for layer in jax_params['h']:
                        if 'c_attn' in name:
                            jax_param = layer['attn']['c_attn'].T
                            break
                        elif 'c_proj' in name and 'attn' in name:
                            jax_param = layer['attn']['c_proj'].T
                            break
                        elif 'c_fc' in name:
                            jax_param = layer['mlp']['c_fc'].T
                            break
                        elif 'c_proj' in name and 'mlp' in name:
                            jax_param = layer['mlp']['c_proj'].T
                            break
                if jax_param is not None:
                    torch_param = param.detach().numpy()
                    param_diff = np.abs(jax_param - torch_param).max()
                    print(f"  Max parameter diff for {name}: {param_diff:.6f}")
        """

        print(f"  Learning rate: {lr:.6f}")
        print()