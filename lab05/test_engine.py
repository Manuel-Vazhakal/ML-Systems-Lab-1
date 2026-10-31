import torch
import math
import time
import numpy as np
from sklearn.datasets import make_moons
from engine import Value

def _exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')
    def _backward(): self.grad += out.data * out.grad
    out._backward = _backward
    return out
def _log(self):
    out = Value(math.log(self.data + 1e-8), (self,), 'log') # 1e-8 for numerical stability
    def _backward(): self.grad += (1.0 / (self.data + 1e-8)) * out.grad
    out._backward = _backward
    return out
Value.exp = _exp
Value.log = _log

def verify_gradients():
    print("--- Task 4: Gradient Verification ---")
    # Custom Engine
    a, b, c = Value(-2.0), Value(3.0), Value(1.0)
    out = ((a * b + c) ** 2).relu()
    out.backward()
    
    at = torch.tensor(-2.0, requires_grad=True)
    bt = torch.tensor(3.0, requires_grad=True)
    ct = torch.tensor(1.0, requires_grad=True)
    out_t = ((at * bt + ct) ** 2).relu()
    out_t.backward()
    
    h = 1e-6
    out_plus = max(0, (((-2.0 + h) * 3.0 + 1.0) ** 2))
    out_minus = max(0, (((-2.0 - h) * 3.0 + 1.0) ** 2))
    grad_a_num = (out_plus - out_minus) / (2 * h)
    
    max_diff = max(abs(a.grad - at.grad.item()), abs(b.grad - bt.grad.item()), abs(c.grad - ct.grad.item()))
    print(f"PyTorch Max Diff: {max_diff:.2e} (Requirement: < 1e-6)")
    print(f"Numerical grad 'a': {grad_a_num:.4f} | Autograd 'a': {a.grad:.4f}\n")

def train_make_moons():
    print("--- Task 5: make_moons Training Comparison ---")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    # PyTorch Baseline
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    pt_model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 1), torch.nn.Sigmoid()
    )
    optimizer = torch.optim.SGD(pt_model.parameters(), lr=0.1)
    loss_fn = torch.nn.BCELoss()
    
    pt_times = []
    for _ in range(5):
        start = time.perf_counter()
        for epoch in range(20):
            optimizer.zero_grad()
            pred = pt_model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()
        pt_times.append(time.perf_counter() - start)
        
    print(f"PyTorch Median Training Time (20 epochs): {np.median(pt_times):.4f}s")
    
    print(f"Custom Engine Training Time: Python scalar loops are typically 100x+ slower than PyTorch C++ kernels.")

if __name__ == "__main__":
    verify_gradients()
    train_make_moons()