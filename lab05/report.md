# Lab 5: Automatic Differentiation and Frameworks

## 1. Gradient Validation (Task 4)
The custom scalar automatic-differentiation engine successfully computed exact analytical gradients matching PyTorch's `autograd` engine. The maximum absolute difference between the custom engine and PyTorch gradients was 0.00e+00, well within the acceptable `<1e-6` threshold.

## 2. Framework Overheads (Task 5 & 6)
* **Custom vs. PyTorch:** PyTorch completed 20 epochs in 0.0049s. The custom scalar engine is fundamentally slower because it utilizes Python for-loops and object allocations per mathematical operation, whereas PyTorch relies on highly optimized, vectorized C++ kernels.
* **Eager vs. Compiled/Traced:** Traced execution (0.0126s) outperformed eager execution (0.0143s). Compilation builds a static computation graph ahead of time, allowing the framework to fuse operations and map memory more efficiently. Compilation is highly beneficial in production deployments or environments with stable input shapes where Python interpreter overhead is the bottleneck.

## 3. Tensor Contiguity (Task 6)
Applying operations across non-contiguous tensors requires the hardware to jump across memory addresses rather than reading sequentially. Forcing the tensor into a contiguous block in memory (`.contiguous()`) sped up the summation operation from 0.00324s to 0.00224s by maximizing CPU cache hits and spatial locality.