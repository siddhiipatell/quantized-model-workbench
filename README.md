# Quantized Model Workbench
A research and experimentation repository containing Jupyter notebooks for low-bit quantization of machine learning and large language models, focused on efficient inference, memory optimization, and performance analysis.

## [4-Bit Quantized Llama3.1 NemoGuard 8B Content Safety](https://github.com/siddhiipatell/quantized-model-workbench/blob/main/Quantized_4_bit_llama_3_1_nemoguard_8b_content_safety.ipynb)
This notebook demonstrates low-bit (4-bit) quantization of the LLaMA 3.1 NemoGuard-8B large language model for content safety and moderation inference, with an emphasis on reducing GPU memory usage while preserving inference quality.

The workflow is designed for efficient inference in constrained environments such as consumer GPUs or cloud cost-optimized deployments.

### Quantization Methodology
- Quantization Type: Post-training weight-only quantization
- Precision: 4-bit (INT4 / NF4)
- Quantization Strategy:
  - Model weights are quantized while activations remain in higher precision
  - Computation is performed using mixed precision to balance performance and stability
- Objective:
  - Minimize VRAM consumption
  - Enable inference on lower-memory GPUs
  - Maintain acceptable safety classification accuracy

#### Quantization & Optimization
```
bitsandbytes
```
- Enables 4-bit and 8-bit quantized weight loading
- Supports NF4 and FP4 quantization formats
- Provides memory-efficient linear layers for LLM inference

### Quantization Configuration (Conceptual)
```
load_in_4bit = True
bnb_4bit_compute_dtype = torch.float16 / torch.bfloat16
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = True
```
These settings allow efficient low-bit inference while preserving numerical stability.

### Model Architecture
- Base Model: LLaMA 3.1 NemoGuard-8B
- Task: Content safety/policy compliance classification
- Inference Mode: Autoregressive generation with safety-focused prompting
- Optimization Target: Inference-only (no fine-tuning)

### Performance Considerations
- Memory Reduction: ~60–75% VRAM reduction compared to FP16
- Latency: Slight overhead from dequantization during inference
- Accuracy Impact: Minimal degradation for safety classification tasks
