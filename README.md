# About:

This README file contains concepts from torchtune toturial [here](https://pytorch.org/torchtune/stable/overview.html)


# Notes:

> ### Gradient Accumulation
> * **What it is:** <font color='red'>**`Gradient accumulation` is a technique used to simulate a larger batch size when your hardware cannot handle large batches due to memory constraints.**</font> Instead of updating the model's weights after processing a single batch, gradients are accumulated over multiple smaller batches. Once the accumulated gradients match the desired effective batch size, the weights are updated.
This shows why pytorch by default accumulate gradient and you must at the start of the batch call `optimizer.zero_grad()`.
> * **Why it’s useful:** It allows you to train models with limited GPU memory by breaking down large batches into smaller chunks. This is particularly helpful for modest hardware setups.
> * **How it differs from LoRA:** `LoRA` is a **parameter-efficient fine-tuning** method that reduces the number of trainable parameters, while `gradient accumulation` is a **training optimization technique** that helps manage memory usage during training. They serve different purposes but can be used together.

> ### Reduced Precision Training
> * **What it is:** Reduced precision training involves using lower-precision data types (e.g., 16-bit floating point, or FP16) instead of the standard 32-bit floating point (FP32) for computations. This reduces memory usage and speeds up training.
> * **Why it’s useful:** Lower precision requires less memory and can significantly speed up training and inference, especially on GPUs that support it (like NVIDIA GPUs with Tensor Cores).
> * **How it differs from LoRA:** `Reduced precision training` is a **computational optimization technique**, while `LoRA` is a **model optimization technique**. LoRA reduces the number of trainable parameters, whereas reduced precision training reduces the memory and computational cost of the existing parameters.

> ### LoRA (Low-Rank Adaptation)
> * **What it is:** LoRA is a parameter-efficient fine-tuning method that introduces small, trainable low-rank matrices into the model while keeping the original model weights frozen. This allows fine-tuning with significantly fewer parameters.
> * **Why it’s useful:** It reduces memory and computational requirements, making fine-tuning feasible on modest hardware.    
> * **How it differs from `gradient accumulation` and `reduced precision training`:** `LoRA` modifies the model architecture to reduce trainable parameters, while the other two techniques optimize the training process itself.
