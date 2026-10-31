import torch
import time

def execution_and_contiguity():
    print("--- Task 6: Execution Modes & Contiguity ---")
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 1), torch.nn.Sigmoid()
    )
    dummy_input = torch.randn(10000, 2)
    
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Warmup
    for _ in range(10): 
        model(dummy_input)
        traced_model(dummy_input)
    
    start = time.perf_counter()
    for _ in range(100): model(dummy_input)
    eager_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(100): traced_model(dummy_input)
    traced_time = time.perf_counter() - start
    
    print(f"Eager Time: {eager_time:.4f}s | Traced/Compiled Time: {traced_time:.4f}s")
    
    tensor = torch.randn(10000, 1000)
    non_contiguous = tensor.transpose(0, 1)
    contiguous = non_contiguous.contiguous()
    
    print(f"\nNon-contiguous shape: {non_contiguous.shape}, is_contiguous: {non_contiguous.is_contiguous()}")
    print(f"Contiguous shape: {contiguous.shape}, is_contiguous: {contiguous.is_contiguous()}")
    
    start = time.perf_counter()
    non_contiguous.sum(dim=1)
    nc_time = time.perf_counter() - start
    
    start = time.perf_counter()
    contiguous.sum(dim=1)
    c_time = time.perf_counter() - start
    
    print(f"Non-contiguous sum time: {nc_time:.5f}s | Contiguous sum time: {c_time:.5f}s")

if __name__ == "__main__":
    execution_and_contiguity()