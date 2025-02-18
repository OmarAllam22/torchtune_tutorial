# About:

This README file contains concepts from torchtune toturial [**HERE**](https://pytorch.org/torchtune/stable/overview.html)
or [**HERE**](https://pytorch.org/torchtune/stable/index.html#getting-started)
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
> #### **`QLoRA`** is an extension of LoRA that incorporates quantization.
> * It fine-tunes models using low-rank adaptations while keeping the base model quantized (e.g., 4-bit precision).
> * **QLoRA further reduces memory and computational requirements compared to LoRA**, making it possible to fine-tune very large models on consumer-grade hardware. <font color='red'>**Quantizaion is not used only for training but inference**</font>. 

> ### DPO (Direct Preference Optimization)
> * **`DPO (Direct Preference Optimization)`** is **a method for fine-tuning models to align their outputs with human preferences.**
> * Traditional fine-tuning methods often rely on **`reinforcement learning from human feedback (RLHF)`**, which involves **training a reward model** and then using it to guide the fine-tuning process. However, RLHF can be complex and computationally expensive.
> * **DPO simplifies** this process by directly optimizing the model to prefer outputs that align with human preferences, **without needing a separate reward model**.
>> DPO works by comparing pairs of model outputs and optimizing the model to prefer the output that aligns better with human preferences. Here’s how it works:
>> 1. **Data Collection:**
<br> Collect a dataset of paired comparisons, where humans have ranked two model outputs for the same input (e.g., "Output A is better than Output B").
>> 2. **Loss Function:**
<br>DPO uses a preference-based loss function to directly optimize the model. The loss function encourages the model to assign higher probabilities to the preferred outputs and lower probabilities to the less preferred ones. The loss function is designed to maximize the likelihood of the preferred outputs while minimizing the likelihood of the less preferred ones.
>> 3. **Training:**
<br>The model is fine-tuned using the preference data and the DPO loss function. This aligns the model's outputs with human preferences without requiring a separate reward model.

> ### PPO (Proximal Policy Optimization)
> It is different from DPO. <font color='red'>**DPO (Direct Preference Optimization) is not a method in RLHF but PPO is.**</font>
> * PPO (Proximal Policy Optimization) is a popular reinforcement learning algorithm used to train agents (such as language models) to make decisions by optimizing their policies. It is widely used in tasks like fine-tuning large language models (LLMs) with Reinforcement Learning from Human Feedback (RLHF).
> * PPO is a reinforcement learning algorithm designed to optimize the behavior of an agent (e.g., a language model) by maximizing a reward signal.
> * It is called "proximal" because it ensures that updates to the agent's policy are not too large, preventing instability during training.
> * PPO is a policy gradient method, meaning it directly optimizes the policy (the strategy the agent uses to make decisions) rather than learning a value function

> ### Model Distillation:
> Knowledge distillation is a technique used to transfer knowledge from a large, complex model (called the teacher) to a smaller, simpler model (called the student). The goal is to improve the student model’s performance by leveraging the teacher’s learned representations, even though the student has fewer parameters and less capacity.