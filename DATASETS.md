# Training Datasets: Guide

This guide explains how to structure datasets and configure training for all task types supported by soen_toolkit. After reading this, you should be able to create HDF5 datasets for any of the supported paradigms.

## Quick Reference: Supported Task Types

| Paradigm | Mapping | Task Type | Input Shape | Target Shape | Example Use Case |
|----------|---------|-----------|-------------|--------------|------------------|
| Supervised | seq2static | Classification | `[N, T, D]` | `[N]` | Pulse detection, sentiment analysis |
| Supervised | seq2static | Regression | `[N, T, D]` | `[N, K]` | Sequence summarization |
| Supervised | seq2seq | Classification | `[N, T, D]` | `[N, T]` | Sequence labeling, POS tagging |
| Supervised | seq2seq | Regression | `[N, T, D]` | `[N, T, K]` | Time series forecasting |
| Self-supervised | seq2seq | Reconstruction | `[N, T, D]` | (uses inputs) | Autoencoder, denoising |
| Self-supervised | seq2static | Summary Learning | `[N, T, D]` | `[N, K]` | Learning sequence statistics |

## Dataset Structure

### Input Data (Always Required)
- **HDF5 key**: `data`
- **Shape**: `[N, T, D]` where:
  - `N`: number of samples
  - `T`: sequence length (timesteps)
  - `D`: feature dimension
- **Dtype**: `float32`

### Target Data (Labels)
- **HDF5 key**: `labels`
- **Shape and dtype depend on task type** (see table above)

## Task Type Guide

### 1. Supervised seq2static Classification
**Use case**: Classify entire sequences (e.g., "does this sequence contain an anomaly?")

```python
import h5py
import numpy as np

# Example: Pulse detection dataset
N, T, D = 1000, 64, 1  # 1000 samples, 64 timesteps, 1 feature
num_classes = 3  # 0=no pulse, 1=single pulse, 2=double pulse

# Generate synthetic data
data = np.random.randn(N, T, D).astype(np.float32)
labels = np.random.randint(0, num_classes, size=(N,), dtype=np.int64)

# Add some pulses to make it realistic
for i in range(N):
    if labels[i] == 1:  # Single pulse
        pulse_pos = np.random.randint(10, T-10)
        data[i, pulse_pos:pulse_pos+5, 0] += 2.0
    elif labels[i] == 2:  # Double pulse  
        pos1 = np.random.randint(5, T//2)
        pos2 = np.random.randint(T//2, T-5)
        data[i, pos1:pos1+3, 0] += 1.5
        data[i, pos2:pos2+3, 0] += 1.5

# Save to HDF5
with h5py.File("pulse_classification.h5", "w") as f:
    train_group = f.create_group("train")
    train_group.create_dataset("data", data=data[:800])
    train_group.create_dataset("labels", data=labels[:800])
    
    val_group = f.create_group("val") 
    val_group.create_dataset("data", data=data[800:900])
    val_group.create_dataset("labels", data=labels[800:900])
    
    test_group = f.create_group("test")
    test_group.create_dataset("data", data=data[900:])
    test_group.create_dataset("labels", data=labels[900:])
```

**YAML config**:
```yaml
training:
  paradigm: supervised
  mapping: seq2static
  losses: [{name: cross_entropy, weight: 1.0, params: {}}]

data:
  data_path: pulse_classification.h5
  num_classes: 3
  target_seq_len: 64

model:
  time_pooling: {name: max, params: {scale: 1.0}}
```

### 2. Supervised seq2static Regression
**Use case**: Predict a single value from a sequence (e.g., "what's the average temperature?")

```python
# Example: Predict sequence statistics
N, T, D, K = 800, 50, 3, 2  # Predict mean and std

data = np.random.randn(N, T, D).astype(np.float32)
# Target: mean and std of first feature across time
labels = np.column_stack([
    np.mean(data[:, :, 0], axis=1),  # mean
    np.std(data[:, :, 0], axis=1)    # std  
]).astype(np.float32)

with h5py.File("sequence_stats.h5", "w") as f:
    f.create_dataset("data", data=data)
    f.create_dataset("labels", data=labels)
```

### 3. Supervised seq2seq Classification  
**Use case**: Label each timestep (e.g., part-of-speech tagging)

```python
# Example: Sequence labeling
N, T, D = 600, 32, 4
num_classes = 5

data = np.random.randn(N, T, D).astype(np.float32)
# Labels for each timestep - shape [N, T]
labels = np.random.randint(0, num_classes, size=(N, T), dtype=np.int64)

with h5py.File("sequence_labeling.h5", "w") as f:
    f.create_dataset("data", data=data)
    f.create_dataset("labels", data=labels)  # Note: 2D shape [N, T]
```

**Key point**: For seq2seq classification, targets are 2D `[N, T]` not 1D!

### 4. Supervised seq2seq Regression
**Use case**: Time series forecasting, sequence-to-sequence prediction

```python
# Example: Time series forecasting
N, input_len, forecast_len, D = 500, 24, 8, 3

# Input sequences
data = np.random.randn(N, input_len, D).astype(np.float32)
# Add some trend and seasonality
for i in range(N):
    trend = np.linspace(0, np.random.uniform(-1, 1), input_len)
    data[i, :, 0] += trend

# Forecast targets - shape [N, forecast_len, D]
labels = np.random.randn(N, forecast_len, D).astype(np.float32)
# Continue the trend
for i in range(N):
    last_val = data[i, -1, 0]
    trend_continuation = np.linspace(last_val, last_val + np.random.uniform(-0.5, 0.5), forecast_len)
    labels[i, :, 0] = trend_continuation

with h5py.File("forecasting.h5", "w") as f:
    f.create_dataset("data", data=data)
    f.create_dataset("labels", data=labels)  # Shape: [N, forecast_len, D]
```

**YAML config**:
```yaml
training:
  paradigm: supervised
  mapping: seq2seq
  losses: [{name: mse, weight: 1.0, params: {}}]

data:
  data_path: forecasting.h5
  target_seq_len: 8  # Must match forecast_len!
```

### 5. Self-supervised seq2seq Reconstruction
**Use case**: Autoencoders, denoising, sequence reconstruction

```python
# Example: Autoencoder dataset
N, T, D = 1000, 40, 6

data = np.random.randn(N, T, D).astype(np.float32)
# Add some structure/patterns
for i in range(N):
    # Add periodic patterns
    t = np.arange(T)
    data[i, :, 0] += np.sin(2 * np.pi * t / 10)
    data[i, :, 1] += np.cos(2 * np.pi * t / 15)

# For unsupervised, labels can be dummy or omitted
dummy_labels = np.zeros(N, dtype=np.int64)

with h5py.File("autoencoder.h5", "w") as f:
    f.create_dataset("data", data=data)
    f.create_dataset("labels", data=dummy_labels)  # Will be ignored
```

**YAML config**:
```yaml
training:
  paradigm: self_supervised
  mapping: seq2seq  
  losses: [{name: mse, weight: 1.0, params: {}}]

data:
  data_path: autoencoder.h5
  target_seq_len: 40
```

### 6. Self-supervised seq2static Summary Learning
**Use case**: Learn to predict sequence summaries derived from the input data

```python
# Example: Learn sequence statistics in self-supervised way
N, T, D, summary_dim = 600, 30, 5, 3

data = np.random.randn(N, T, D).astype(np.float32)

# Create "labels" that are derived from inputs (e.g., statistics)
# The model learns to predict these from the sequences
labels = np.column_stack([
    np.mean(data[:, :, 0], axis=1),    # mean of first feature
    np.max(data[:, :, 1], axis=1),     # max of second feature  
    np.var(data[:, :, 2], axis=1)      # variance of third feature
]).astype(np.float32)

with h5py.File("unsupervised_summary.h5", "w") as f:
    f.create_dataset("data", data=data)
    f.create_dataset("labels", data=labels)
```

**YAML config**:
```yaml
training:
  paradigm: self_supervised
  mapping: seq2static
  losses: [{name: mse, weight: 1.0, params: {}}]

data:
  data_path: unsupervised_summary.h5
  target_seq_len: 30

model:
  time_pooling: {name: mean, params: {scale: 1.0}}
```

## HDF5 File Layouts

### Option 1: Pre-split (Recommended)
```
dataset.h5/
├── train/
│   ├── data: [N_train, T, D]
│   └── labels: [depends on task]
├── val/
│   ├── data: [N_val, T, D] 
│   └── labels: [depends on task]
└── test/
    ├── data: [N_test, T, D]
    └── labels: [depends on task]
```

### Option 2: Single dataset (auto-split)
```
dataset.h5/
├── data: [N_total, T, D]
└── labels: [depends on task]
```

## Loss Function Selection

Choose appropriate loss functions based on your task type:

### Classification Tasks
```yaml
training:
  losses: [{name: cross_entropy, weight: 1.0, params: {}}]
```
- **cross_entropy**: Standard for classification (both seq2static and seq2seq)
- **gap_loss**: Additional regularization for better class separation (params: {margin: 0.2})
- **top_gap_loss**: Variant of gap loss focusing on top predictions
- **rich_margin_loss**: Advanced margin-based loss for classification

### Regression Tasks
```yaml
training:
  losses: [{name: mse, weight: 1.0, params: {}}]
```
- **mse**: Mean Squared Error - most common for regression
- **mse_gradient_cutoff**: MSE with gradient cutoff for stability

### Self-supervised Tasks
```yaml
training:
  losses: [{name: mse, weight: 1.0, params: {}}]
```
- **mse**: Standard for reconstruction tasks
- **autoregressive_loss**: For sequence generation tasks
- **autoregressive_cross_entropy**: Cross-entropy for autoregressive tasks

### Specialized Loss Functions
Available for specific use cases:
- **reg_J_loss**: Regularization on connection weights
- **get_off_the_ground_loss**: Encourages non-zero activations
- **exp_high_state_penalty**: Penalizes extremely high activations
- **gravity_quantization_loss**: For quantization-aware training
- **branching_loss**: For tree-like model structures
- **final_timestep_zero_mse**: MSE on final timestep only

### Multiple Loss Functions
Combine multiple losses with different weights:
```yaml
training:
  losses:
    - {name: cross_entropy, weight: 1.0, params: {}}
    - {name: gap_loss, weight: 0.3, params: {margin: 0.2}}
```

## Time Pooling Methods

For seq2static tasks, configure how sequences are reduced to single outputs:

```yaml
model:
  time_pooling:
    name: final        # Use last timestep
    # name: mean       # Average over time  
    # name: max        # Max over time
    # name: mean_last_n # Average last n timesteps
    # params: {n: 5}
    params: {scale: 1.0}
```

## Common Issues and Solutions

### ❌ Sequence Length Mismatch (seq2seq)
```
RuntimeError: Expected size [8, 16], got [8, 24]
```
**Solution**: Set `data.target_seq_len` to match your label sequence length.

### ❌ Wrong Label Dtype
```
RuntimeError: Expected floating point type for target but got Long
```
**Solution**: Use `np.int64` for classification, `np.float32` for regression.

### ❌ Wrong Label Shape for seq2seq Classification
```
RuntimeError: 0D or 1D target tensor expected, multi-target not supported
```
**Solution**: seq2seq classification needs 2D targets `[N, T]`, not 1D `[N]`.

## Validation Script

Use this to check your dataset:

```python
import h5py

def validate_dataset(path, task_type):
    with h5py.File(path, 'r') as f:
        # Check structure
        if 'train' in f:
            data_shape = f['train/data'].shape
            label_shape = f['train/labels'].shape
        else:
            data_shape = f['data'].shape  
            label_shape = f['labels'].shape
            
        print(f"Data shape: {data_shape}")
        print(f"Label shape: {label_shape}")
        
        # Validate shapes based on task
        N, T, D = data_shape
        
        if task_type == "supervised_seq2static_classification":
            assert label_shape == (N,), f"Expected [N], got {label_shape}"
        elif task_type == "supervised_seq2seq_classification":  
            assert label_shape == (N, T), f"Expected [N, T], got {label_shape}"
        elif task_type == "supervised_seq2static_regression":
            assert len(label_shape) == 2, f"Expected [N, K], got {label_shape}"
        elif task_type == "supervised_seq2seq_regression":
            assert len(label_shape) == 3, f"Expected [N, T, K], got {label_shape}"
            
        print(f"✅ Dataset valid for {task_type}")

# Example usage
validate_dataset("pulse_classification.h5", "supervised_seq2static_classification")
```

## Advanced Features

### One-hot Input Encoding
For token-based inputs:
```yaml
data:
  input_encoding: "one_hot"
  vocab_size: 1000
```
Then `data` should contain integer token indices `[N, T]` or `[N, T, 1]`.

### Multiple Loss Functions
```yaml
training:
  losses:
    - {name: cross_entropy, weight: 1.0, params: {}}
    - {name: gap_loss, weight: 0.3, params: {margin: 0.2}}
```

## Summary

The key to creating datasets is understanding the **target shapes**:
- **Classification**: Integer labels (int64)
- **Regression**: Float labels (float32)  
- **seq2static**: Single output per sequence
- **seq2seq**: Output for each timestep
- **Self-supervised**: Labels ignored or derived from inputs (use inputs as targets)

With these patterns, you can create datasets for any task your soen_toolkit models need to solve!

## Where to Look in Code

- Data loaders: `src/soen_toolkit/training/data/dataloaders.py`
- DataModule: `src/soen_toolkit/training/data/data_module.py`
- Lightning wrapper: `src/soen_toolkit/training/models/lightning_wrapper.py`
- Test examples: `src/soen_toolkit/tests/training/test_comprehensive_training_pipeline.py`
