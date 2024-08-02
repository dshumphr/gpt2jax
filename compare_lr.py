import jax
import jax.numpy as jnp
import torch
import numpy as np
from model import Transformer
from tmodel import GPT, GPTConfig
import optax
from train_gpt2 import compute_loss_and_grads
from torch.nn import functional as F
import math

# Assuming you have the JAX and PyTorch models and related functions defined
heads = 12
layers = 12
hidden_size = 768
vocab_size = 50304
B = 4
L = 1024
max_steps = 19073

# Learning rate scheduler using optax
warmup_steps = 715
max_lr = 6e-4
min_lr = max_lr * 0.1
init_lr = max_lr / warmup_steps
schedule_fn = optax.warmup_cosine_decay_schedule(
    init_value=init_lr,
    peak_value=max_lr,
    warmup_steps=warmup_steps,
    decay_steps=max_steps,
    end_value=min_lr
)

# LR function for pytorch
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def test_lr_schedule():
    # Compare learning rate schedules
    for step in range(1000, max_steps+100, 100):
        jax_lr = schedule_fn(step)
        torch_lr = get_lr(step)
        print(f"Step {step}:")
        print(f"JAX LR: {jax_lr}")
        print(f"PyTorch LR: {torch_lr}")
        print(f"Difference: {abs(jax_lr - torch_lr)}")
        print()

test_lr_schedule()