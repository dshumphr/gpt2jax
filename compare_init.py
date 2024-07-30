import jax
import jax.numpy as jnp
import torch
import numpy as np
from collections import defaultdict
from model import Transformer
from tmodel import GPT, GPTConfig

def analyze_init_distribution(name, tensor):
    """Analyze the distribution of a given tensor."""
    if isinstance(tensor, jnp.ndarray):
        tensor = np.array(tensor)
    elif isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    return {
        'name': name,
        'shape': tensor.shape,
        'mean': np.mean(tensor),
        'std': np.std(tensor),
        'min': np.min(tensor),
        'max': np.max(tensor)
    }

def print_init_info(info):
    """Print the initialization info in a formatted way."""
    print(f"Parameter: {info['name']}")
    print(f"  Shape: {info['shape']}")
    print(f"  Mean: {info['mean']:.6f}")
    print(f"  Std Dev: {info['std']:.6f}")
    print(f"  Min: {info['min']:.6f}")
    print(f"  Max: {info['max']:.6f}")
    print()

def compare_init_distributions(jax_model, torch_model):
    print("JAX Model Initialization:")
    jax_info = defaultdict(list)
    
    def traverse_jax_params(params, prefix=''):
        if isinstance(params, dict):
            for k, v in params.items():
                traverse_jax_params(v, prefix + k + '.')
        elif isinstance(params, list):
            for i, v in enumerate(params):
                traverse_jax_params(v, prefix + f'[{i}].')
        else:
            info = analyze_init_distribution(prefix[:-1], params)
            print_init_info(info)
            jax_info[info['shape']].append(info)

    traverse_jax_params(jax_model)

    print("\nPyTorch Model Initialization:")
    torch_info = defaultdict(list)
    
    for name, param in torch_model.named_parameters():
        info = analyze_init_distribution(name, param)
        print_init_info(info)
        torch_info[info['shape']].append(info)
    
    print("\nComparison Summary:")
    all_shapes = set(list(jax_info.keys()) + list(torch_info.keys()))
    for shape in all_shapes:
        print(f"Shape: {shape}")
        if shape in jax_info and shape in torch_info:
            jax_mean_std = np.mean([info['std'] for info in jax_info[shape]])
            torch_mean_std = np.mean([info['std'] for info in torch_info[shape]])
            print(f"  JAX mean std: {jax_mean_std:.6f}")
            print(f"  PyTorch mean std: {torch_mean_std:.6f}")
            print(f"  Relative difference: {abs(jax_mean_std - torch_mean_std) / max(jax_mean_std, torch_mean_std):.2%}")
        else:
            print("  Shape mismatch between JAX and PyTorch models")
        print()

# Usage
heads = 12
layers = 12
hidden_size = 768
vocab_size = 50304
B = 1
L = 1024
max_steps = 19073

key = jax.random.PRNGKey(0)
jax_model = Transformer.init(key, vocab_size, heads, hidden_size, layers, L)
torch_model = GPT(GPTConfig(vocab_size=50304))
compare_init_distributions(jax_model, torch_model)