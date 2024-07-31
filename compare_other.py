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

def test_loss_calculation():
    # Generate random data
    batch_size = 4
    seq_length = 1024
    vocab_size = 50304
    
    # JAX
    jax_batch = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_length), 0, vocab_size)
    jax_logits = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_length-1, vocab_size))
    
    # PyTorch
    torch_batch = torch.from_numpy(np.array(jax_batch))
    torch_logits = torch.from_numpy(np.array(jax_logits))
    
    # Calculate loss in JAX
    jax_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(jax_logits, jax_batch[:, 1:]))
    
    # Calculate loss in PyTorch
    torch_loss = F.cross_entropy(torch_logits.view(-1, vocab_size), torch_batch[:, 1:].reshape(-1).long())
    
    print(f"JAX loss: {jax_loss}")
    print(f"PyTorch loss: {torch_loss.item()}")
    print(f"Difference: {abs(jax_loss - torch_loss.item())}")

def test_grad_calculation():
    # Initialize models with the same parameters
    jax_params = Transformer.init(jax.random.PRNGKey(0), vocab_size, heads, hidden_size, layers, L)
    torch_model = GPT(GPTConfig(vocab_size=vocab_size))
    
    # Generate random batch
    batch_size = 4
    seq_length = 1024
    jax_batch = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_length), 0, vocab_size)
    torch_batch = torch.from_numpy(np.array(jax_batch))
    
    # JAX forward and backward
    jax_loss, jax_grads = compute_loss_and_grads(jax_params, jax_batch)
    
    # PyTorch forward and backward
    torch_logits, torch_loss = torch_model(torch_batch[:, :-1], torch_batch[:, 1:])
    torch_loss.backward()
    
    # Compare gradients
    for (jax_name, jax_grad), (torch_name, torch_param) in zip(
        jax.tree_util.tree_leaves_with_path(jax_grads),
        torch_model.named_parameters()
    ):
        torch_grad = torch_param.grad.numpy()
        print(f"Layer: {jax_name}")
        print(f"Max difference: {np.max(np.abs(jax_grad - torch_grad))}")

def test_optimizer_step():
    # Initialize models and optimizers
    jax_params = Transformer.init(jax.random.PRNGKey(0), vocab_size, heads, hidden_size, layers, L)
    jax_optimizer = optax.adamw(learning_rate=6e-4, b1=0.9, b2=0.95, weight_decay=0.1)
    jax_opt_state = jax_optimizer.init(jax_params)
    
    torch_model = GPT(GPTConfig(vocab_size=vocab_size))
    torch_optimizer = torch.optim.AdamW(torch_model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Generate random gradients
    jax_grads = jax.tree_map(lambda x: jax.random.normal(jax.random.PRNGKey(0), x.shape), jax_params)
    torch_grads = [torch.from_numpy(np.array(g)) for g in jax.tree_util.tree_leaves(jax_grads)]
    
    # Apply gradients
    jax_updates, jax_opt_state = jax_optimizer.update(jax_grads, jax_opt_state, jax_params)
    jax_new_params = optax.apply_updates(jax_params, jax_updates)
    
    for param, grad in zip(torch_model.parameters(), torch_grads):
        param.grad = grad
    torch_optimizer.step()
    
    # Compare updated parameters
    for (jax_name, jax_param), (torch_name, torch_param) in zip(
        jax.tree_util.tree_leaves_with_path(jax_new_params),
        torch_model.named_parameters()
    ):
        print(f"Layer: {jax_name}")
        print(f"Max difference: {np.max(np.abs(jax_param - torch_param.detach().numpy()))}")

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

# Run the tests
#test_grad_calculation()
#test_optimizer_step()
test_lr_schedule()