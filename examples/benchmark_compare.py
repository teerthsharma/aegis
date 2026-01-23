import time
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Python/Traditional ML Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
#
# This script simulates a traditional fixed-epoch training loop to compare
# against AEGIS's topological convergence (Seal Loop).
#
# ═══════════════════════════════════════════════════════════════════════════════

def simple_linear_regression():
    # Synthetic data
    X = np.array([1.0, 1.2, 3.5, 3.7, 7.1, 7.3, 10.0, 10.2, 15.5, 15.6])
    y = 2.0 * X + 1.0 + np.random.normal(0, 0.5, len(X))
    
    # Model parameters
    w = 0.0
    b = 0.0
    lr = 0.01
    epochs = 10000 # Traditional fixed epochs
    
    start_time = time.time_ns()
    
    # Traditional Loop: Run for fixed N epochs regardless of convergence
    for epoch in range(epochs):
        # Forward pass
        y_pred = w * X + b
        
        # Loss (MSE)
        loss = np.mean((y - y_pred)**2)
        
        # Backward pass (Gradient Descent)
        dw = -2 * np.mean(X * (y - y_pred))
        db = -2 * np.mean(y - y_pred)
        
        # Update
        w -= lr * dw
        b -= lr * db
        
        # Check convergence (usually manual threshold, but loop continues in simple scripts)
        # In real PyTorch/TF scripts, people often just set epochs=1000
    
    end_time = time.time_ns()
    duration_ms = (end_time - start_time) / 1_000_000
    
    print(f"Python (Fixed {epochs} Epochs): {duration_ms:.3f} ms")
    print(f"Final Loss: {loss:.6f}")

if __name__ == "__main__":
    print("-" * 40)
    print("Running Python Benchmark...")
    print("-" * 40)
    simple_linear_regression()
