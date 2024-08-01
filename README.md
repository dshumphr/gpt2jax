# nanoGPT-JAX

This repository contains a from-scratch implementation of nanoGPT using JAX (implemented in order to learn, so using vanilla JAX as much as possible). It is based on Andrej Karpathy's GPT-2 from scratch video series.

Note: This is a work in progress and doesn't yet implement everything from the video series.

## Overview

The goal of this project is to reproduce the GPT-2 (124M) model using JAX, starting from an empty file and gradually building up the implementation. The code can also be extended to reproduce larger GPT-3 models with sufficient computational resources.

## Features

- Implements GPT-2 (124M) architecture using JAX
- Reproduces the training process of the original GPT-2 model
- Can be scaled to larger models (e.g., GPT-3) with appropriate hardware

## Requirements

- JAX
- Cloud GPU (recommended for faster training)

## Usage

TODO

## Training

TODO

## Gaps between this and Karpathy implementation
- Mixed precision (although jax already does matmuls with tf32 by default on gpu)
- Fused Adam
- Reduced batch size due to my own hardware limits

Note AI.md is added to enable a sonnet extension (https://github.com/dshumphr/tad)