import jax
import jax.numpy as jnp
import optax
import torch
import numpy as np
import math

# Shared hyperparameters
input_dim = 10
hidden_dim = 5
output_dim = 1
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
init_lr = max_lr / warmup_steps
max_steps = 19000
weight_decay = 0.1
B = 16  # batch size
L = 32  # sequence length
accumulation_steps = 32

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, L, input_dim).astype(np.float32)
y = np.sum(X, axis=(1, 2), keepdims=True).astype(np.float32)

# JAX implementation
def init_jax_params(key):
    k1, k2 = jax.random.split(key)
    return {
        'w1': jax.random.normal(k1, (input_dim, hidden_dim)),
        'w2': jax.random.normal(k2, (hidden_dim, output_dim)),
    }

def jax_forward(params, x):
    h = jnp.dot(x, params['w1'])
    return jnp.dot(h, params['w2'])

@jax.jit
def jax_loss_fn(params, x, y):
    y_pred = jax_forward(params, x)
    return jnp.mean((y_pred - y) ** 2)

@jax.jit
def jax_compute_loss_and_grads(params, batch):
    def loss_fn(params):
        return jax_loss_fn(params, batch[0], batch[1])
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

@jax.jit
def jax_update_params(params, grads, optimizer_state):
    updates, optimizer_state = jax_optimizer.update(grads, optimizer_state, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state

# PyTorch implementation
class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)

# Learning rate schedule
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Initialize models
jax_key = jax.random.PRNGKey(42)
jax_params = init_jax_params(jax_key)

torch_model = TorchMLP()
torch_model.fc1.weight.data = torch.tensor(jax_params['w1'].T.tolist())
torch_model.fc2.weight.data = torch.tensor(jax_params['w2'].T.tolist())
torch_model.fc1.bias.data.zero_()
torch_model.fc2.bias.data.zero_()

# JAX optimizer
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

# PyTorch optimizer
torch_optimizer = torch.optim.AdamW(torch_model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=weight_decay)

# Training loop
for step in range(max_steps):
    jax_loss_accum = 0.0
    torch_loss_accum = 0.0

    for _ in range(accumulation_steps):
        # Sample batch
        idx = np.random.choice(len(X), B)
        x_batch, y_batch = X[idx], y[idx]

        # JAX update
        jax_loss, jax_grads = jax_compute_loss_and_grads(jax_params, (x_batch, y_batch))
        jax_loss_accum += jax_loss

        # PyTorch update
        torch_x = torch.tensor(x_batch)
        torch_y = torch.tensor(y_batch)
        torch_y_pred = torch_model(torch_x.reshape(-1, input_dim)).reshape(B, L, -1)
        torch_loss = torch.mean((torch_y_pred - torch_y) ** 2)
        torch_loss_accum += torch_loss.item()
        (torch_loss / accumulation_steps).backward()

        # JAX optimizer step
        jax_params, jax_opt_state = jax_update_params(jax_params, jax_grads, jax_opt_state)

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
        
        # Compare gradients
        """
        for name, param in torch_model.named_parameters():
            print(name)
            if 'weight' in name:
                jax_grad = jax_grads[f'w{name[2]}'].T
                if param.grad is None:
                    print("None")
                    continue
                torch_grad = param.grad.numpy()
                grad_diff = np.abs(jax_grad - torch_grad).max()
                print(f"  Max gradient diff for {name}: {grad_diff:.6f}")
        """
        """
        # Compare parameters
        for name, param in torch_model.named_parameters():
            print(name)
            if 'weight' in name:
                jax_param = jax_params[f'w{name[2]}'].T
                torch_param = param.detach().numpy()
                param_diff = np.abs(jax_param - torch_param).max()
                print(f"  Max parameter diff for {name}: {param_diff:.6f}")
        """

        print(f"  Learning rate: {lr:.6f}")
        print()