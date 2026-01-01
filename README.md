# Nemotron EN↔VI Translation Fine-tuning

Fine-tuning NVIDIA Nemotron-Nano-9B-v2 for bidirectional English ↔ Vietnamese translation using Unsloth (2-5x faster, 60% less VRAM).

## Requirements

- **GPU:** NVIDIA GPU with 16GB+ VRAM (A100, RTX 4090, RTX 5090)
- **RAM:** 32GB+ recommended
- **Storage:** ~500GB for model + dataset
- **CUDA:** 12.1 or newer

## Performance

| Method | Speed | VRAM | Quality |
|--------|-------|------|---------|
| Standard PEFT | 1x | 16GB+ | ✓ |
| **Unsloth** | 2-5x | 6-10GB | ✓✓ |

---

## Quick Start

### 1. Installation

```bash
cd d:\Pythera\pythera

# Install dependencies
pip install -r requirements.txt

# Install Unsloth (choose based on your CUDA version)
# CUDA 12.1:
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Install mamba-ssm (required for Nemotron)
conda install -c conda-forge mamba-ssm
```

### 2. Dry Run (Test)

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python finetune_nemotron.py --dry_run
```

### 3. Full Training

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python finetune_nemotron.py \
    --train_file ../datasets/dataset/train.json \
    --val_file ../datasets/dataset/val.json \
    --output_dir ../checkpoints/nemotron-translation \
    --num_train_epochs 1
```

---

## Training Options

### Single GPU
```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python finetune_nemotron.py \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

### Multi-GPU
```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python finetune_nemotron.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

### Custom Hyperparameters
```bash
python finetune_nemotron.py \
    --lora_r 64 \                    # LoRA rank
    --lora_alpha 128 \               # LoRA alpha (2x rank)
    --learning_rate 2e-4 \           # Learning rate
    --max_seq_length 512 \           # Max sequence length
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1
```

---

## Dataset

- **Train:** 1,157,897 samples → 2,315,794 after bidirectional processing
- **Validation:** 4,437 samples
- **Test:** 3,622 samples
- **Sources:** mtet (70%) + MedEV (30%)

Each sample is processed TWICE:
1. EN → VI: `Dịch câu sau sang tiếng Việt: "{en}"`
2. VI → EN: `Translate the following sentence into English: "{vi}"`

---

## Key Features

### 1. Unsloth Optimizations
- **FastLanguageModel** for 2-5x faster training
- **Optimized gradient checkpointing**
- **8-bit AdamW optimizer**
- **Automatic model merging**

### 2. Mamba + Transformer Hybrid
LoRA targets both architectures:
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP
    "in_proj", "out_proj",                      # Mamba layers
]
```

### 3. Response-Only Training
Uses `train_on_responses_only` to mask instruction tokens:
```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

### 4. Proper Chat Template
Uses Nemotron's native chat template:
```python
text = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=False
)
```

---

## Inference

### Using Unsloth
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../checkpoints/nemotron-translation/merged",
    max_seq_length=512,
    dtype=None,
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": 'Dịch câu sau sang tiếng Việt: "Hello, how are you?"'}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using Standard Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "../checkpoints/nemotron-translation/merged",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("../checkpoints/nemotron-translation/merged")
```

---

## Monitoring

### WandB
```bash
# Login (first time only)
wandb login

# Training with WandB
python finetune_nemotron.py \
    --use_wandb \
    --wandb_project "nemotron-translation" \
    --wandb_run_name "run-1"
```

### TensorBoard
```bash
tensorboard --logdir ../checkpoints/nemotron-translation
# Access: http://localhost:6006
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python finetune_nemotron.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 256
```

### Unsloth Import Error
```bash
# Set env before running
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune_nemotron.py
```

### mamba-ssm Not Found
```bash
conda install -c conda-forge mamba-ssm
# Or: pip install mamba-ssm --no-build-isolation
```

### Training Slow
```bash
# Increase batch size if VRAM allows
python finetune_nemotron.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

---

## Hyperparameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_r` | 64 | LoRA rank |
| `--lora_alpha` | 128 | LoRA scaling (2x rank) |
| `--lora_dropout` | 0.0 | Dropout (Unsloth recommends 0) |
| `--per_device_train_batch_size` | 2 | Batch size per GPU |
| `--gradient_accumulation_steps` | 8 | Effective batch = 2 × 8 = 16 |
| `--learning_rate` | 2e-4 | Learning rate |
| `--max_seq_length` | 512 | Max sequence length |
| `--num_train_epochs` | 1 | Number of epochs |

---

## Training Time Estimates

| Dataset Size | GPU | Batch Size | Time/Epoch |
|--------------|-----|------------|------------|
| 2.3M samples | A100 (80GB) | 4 | ~12 hours |
| 2.3M samples | A100 (40GB) | 2 | ~24 hours |
| 2.3M samples | RTX 5090 (32GB) | 2 | ~36 hours |

---

## Files

- `finetune_nemotron.py` - Main training script with Unsloth
- `data_collator.py` - NemotronTranslationCollator
- `prepare_dataset.py` - Dataset preprocessing
- `requirements.txt` - Dependencies

---

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Nemotron Model](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [TRL Documentation](https://huggingface.co/docs/trl)
