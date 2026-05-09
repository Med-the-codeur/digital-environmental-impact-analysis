"""
Custom implementation of Scaled Dot-Product Attention
without using PyTorch - using only NumPy
"""

import numpy as np
import math
from codecarbon import OfflineEmissionsTracker


def custom_transpose(matrix, axes=None):
    """
    Custom transpose implementation
    
    Args:
        matrix: numpy array to transpose
        axes: tuple of axes to permute (default: reverse all axes)
    
    Returns:
        Transposed array
    """
    if axes is None:
        # Default transpose: reverse all axes
        axes = tuple(reversed(range(len(matrix.shape))))
    
    # Use numpy's transpose with specified axes
    return np.transpose(matrix, axes)


def custom_matmul(a, b):
    """
    Custom matrix multiplication implementation
    
    Args:
        a: First matrix (numpy array)
        b: Second matrix (numpy array)
    
    Returns:
        Result of matrix multiplication
    
    Raises:
        ValueError: if matrices cannot be multiplied
    """
    # Get shapes
    shape_a = a.shape
    shape_b = b.shape
    
    # Handle different dimensions
    if len(shape_a) == 1 and len(shape_b) == 1:
        # Vector dot product
        if shape_a[0] != shape_b[0]:
            raise ValueError(f"Incompatible dimensions: {shape_a} and {shape_b}")
        return np.sum(a * b)
    
    elif len(shape_a) == 1:
        # Vector-matrix: reshape vector to (1, n)
        a = a.reshape(1, -1)
        result = custom_matmul(a, b)
        return result.reshape(-1) if result.shape[0] == 1 else result
    
    elif len(shape_b) == 1:
        # Matrix-vector: reshape vector to (n, 1)
        b = b.reshape(-1, 1)
        result = custom_matmul(a, b)
        return result.reshape(-1)
    
    elif len(shape_a) == 2 and len(shape_b) == 2:
        # Standard matrix multiplication
        if shape_a[1] != shape_b[0]:
            raise ValueError(
                f"Cannot multiply matrices with shapes {shape_a} and {shape_b}. "
                f"Inner dimensions must match: {shape_a[1]} != {shape_b[0]}"
            )
        
        m, n = shape_a
        n2, p = shape_b
        
        # Initialize result matrix
        result = np.zeros((m, p))
        
        # Perform matrix multiplication
        for i in range(m):
            for j in range(p):
                for k in range(n):
                    result[i, j] += a[i, k] * b[k, j]
        
        return result
    
    elif len(shape_a) >= 2 and len(shape_b) >= 2:
        # Batch matrix multiplication
        # For shape (*, n, m) @ (*, m, p) -> (*, n, p)
        if shape_a[-1] != shape_b[-2]:
            raise ValueError(
                f"Cannot multiply matrices with shapes {shape_a} and {shape_b}. "
                f"Inner dimensions must match: {shape_a[-1]} != {shape_b[-2]}"
            )
        
        # Use numpy's einsum for efficient batch multiplication
        return np.einsum('...ij,...jk->...ik', a, b)
    
    else:
        raise ValueError(
            f"Unsupported shapes for matmul: {shape_a} and {shape_b}"
        )


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled Dot-Product Attention mechanism
    
    Formula: Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
    
    Args:
        q: Query matrix (seq_len, d_k)
        k: Key matrix (seq_len, d_k)
        v: Value matrix (seq_len, d_k)
        mask: Optional mask for attention (seq_len, seq_len)
    
    Returns:
        values: Output of attention (seq_len, d_k)
        attention: Attention weights (seq_len, seq_len)
    """
    with OfflineEmissionsTracker(country_iso_code="TN") as tracker:
        # Get dimension of keys
        d_k = q.shape[-1]
        
        # Compute attention scores: Q @ K^T
        k_transpose = custom_transpose(k, axes=(1, 0))
        attn_logits = custom_matmul(q, k_transpose)
        
        # Scale by sqrt(d_k)
        attn_logits = attn_logits / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            attn_logits = np.where(mask == 0, -1e9, attn_logits)
        
        # Apply softmax to get attention weights
        # Softmax(x) = exp(x) / sum(exp(x))
        # For numerical stability: Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
        exp_logits = np.exp(attn_logits - np.max(attn_logits, axis=-1, keepdims=True))
        attention = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Apply attention to values: Attention @ V
        values = custom_matmul(attention, v)
    
    return values, attention


if __name__ == "__main__":
    # Test parameters
    seq_len, d_k = 300, 200
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create random query, key, value matrices
    q = np.random.randn(seq_len, d_k).astype(np.float32)
    k = np.random.randn(seq_len, d_k).astype(np.float32)
    v = np.random.randn(seq_len, d_k).astype(np.float32)
    
    print("Test Custom Implementation")
    print("=" * 50)
    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    print()
    
    # Test transpose
    print("Testing custom_transpose():")
    k_t = custom_transpose(k, axes=(1, 0))
    print(f"Original K shape: {k.shape}")
    print(f"Transposed K shape: {k_t.shape}")
    print()
    
    # Test matmul
    print("Testing custom_matmul():")
    result = custom_matmul(q, k_t)
    print(f"Q @ K^T shape: {result.shape}")
    print()
    
    # Test scaled dot product attention
    print("Testing scaled_dot_product_attention():")
    print("Starting attention computation...")
    values, attention = scaled_dot_product_attention(q, k, v)
    print("Completed attention computation")
    print(f"Output values shape: {values.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Attention weights sum (should be close to 1): {np.sum(attention[0]):.6f}")
    print()
    
    # Display sample values
    print("Sample values (first 5x5 block):")
    print(values[:5, :5])
    print()
    print("Sample attention weights (first 5x5 block):")
    print(attention[:5, :5])
