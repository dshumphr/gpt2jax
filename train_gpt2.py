import jax
import jax.numpy as jnp
import tiktoken
import optax
from typing import Dict, Any
import time
from flash_attention_jax import causal_flash_attention
import os
from model import Transformer
import pickle
import matplotlib.pyplot as plt
from hellaswag import render_example, iterate_examples

#jax.config.update("jax_debug_nans", True)
def check_nan(tensor, name):
    if jnp.isnan(tensor).any():
        print(f"NaN detected in {name}")
        print(tensor)

@jax.jit
def compute_loss_and_grads(params, x, y):
    def loss_fn(params):
        logits = Transformer.apply(params, jax.nn.one_hot(x, vocab_size))
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
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

def visualize_loss(loss_file_path):
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    hellaswag_steps = []
    hellaswag_accuracies = []

    with open(loss_file_path, 'r') as f:
        for line in f:
            if 'Train Loss' in line:
                parts = line.split(',')
                step = int(parts[0].split()[1])
                loss = float(parts[1].split(':')[1])
                train_steps.append(step)
                train_losses.append(loss)
            elif 'Validation Loss' in line:
                parts = line.split(',')
                step = int(parts[0].split()[1])
                loss = float(parts[1].split(':')[1])
                val_steps.append(step)
                val_losses.append(loss)
            elif 'Hellaswag Accuracy' in line:
                parts = line.split(',')
                step = int(parts[0].split()[1])
                accuracy = float(parts[1].split(':')[1])
                hellaswag_steps.append(step)
                hellaswag_accuracies.append(accuracy)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    ax1.plot(train_steps, train_losses, label='Train Loss')
    ax1.plot(val_steps, val_losses, label='Validation Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(hellaswag_steps, hellaswag_accuracies, label='Hellaswag Accuracy')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Hellaswag Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# Hparams
heads = 12
layers = 12
hidden_size = 768
vocab_size = 50304
B = 16
L = 1024
max_steps = 19073

# Initialize model parameters
key = jax.random.PRNGKey(0)
params = Transformer.init(key, vocab_size, heads, hidden_size, layers, L)

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

# Helper function for hellaswag evaluation
@jax.jit
def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[..., :-1, :]
    shift_tokens = tokens[..., 1:]
    shift_losses = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_tokens)
    shift_mask = mask[..., 1:]
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = jnp.sum(masked_shift_losses, axis=1)
    avg_loss = sum_loss / jnp.sum(shift_mask, axis=1)
    return jnp.argmin(avg_loss)

# Hellaswag evaluation function
def evaluate_hellaswag(params):
    num_correct_norm = 0
    num_total = 0
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        logits = Transformer.apply(params, jax.nn.one_hot(tokens, vocab_size))
        pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    return num_correct_norm / num_total

start_time = time.time()
dataloader = DataLoaderLite(B, L, 'train')
valloader = DataLoaderLite(B, L, 'val')
with open("loss_history.txt", "w") as loss_file:
    for step in range(max_steps):
        for _ in range(accumulation_steps):
            x, y = dataloader.next_batch()
            loss, grads = compute_loss_and_grads(params, x, y)
            accumulated_loss += loss
            params, optimizer_state = update_params(params, grads, optimizer_state)

        # Calculate tokens/s
        tokens_processed = ((step + 1) * tokens_per_batch * accumulation_steps)
        elapsed_time = time.time() - start_time
        tokens_per_second = tokens_processed / elapsed_time

        print(f"Step {step + 1}, Train Loss: {accumulated_loss / accumulation_steps:.4f}, Tokens/s: {tokens_per_second:.2f}")
        loss_file.write(f"Step {step + 1}, Train Loss: {accumulated_loss / accumulation_steps:.4f}, Tokens/s: {tokens_per_second:.2f}\n")
        accumulated_loss = 0.0

        if (step + 1) % 1000 == 0:
            val_loss = 0.0
            for _ in range(accumulation_steps):
                val_batch, _ = valloader.next_batch()
                val_loss_step, _ = compute_loss_and_grads(params, val_batch)
                val_loss += val_loss_step
            val_loss /= accumulation_steps
            print(f"Step {step + 1}, Validation Loss: {val_loss:.4f}")
            loss_file.write(f"Step {step + 1}, Validation Loss: {val_loss:.4f}\n")
            
            # Run hellaswag evaluation
            hellaswag_accuracy = evaluate_hellaswag(params)
            print(f"Step {step + 1}, Hellaswag Accuracy: {hellaswag_accuracy:.4f}")
            loss_file.write(f"Step {step + 1}, Hellaswag Accuracy: {hellaswag_accuracy:.4f}\n")
            
            loss_file.flush()

print("\nTraining completed. Final sample output:")
print(sample(params, 10))

# Save final model
final_checkpoint = {
    'params': params,
    'optimizer_state': optimizer_state
}
with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_checkpoint, f)
print("Final model saved.")

# Visualize loss results
visualize_loss("loss_history.txt")
print("Training metrics plot saved as 'training_metrics.png'")