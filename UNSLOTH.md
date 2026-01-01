# Unsloth Fine-tuning for Nemotron

Faster alternative to standard fine-tuning using Unsloth optimizations.

## Performance

| Method | Speed | VRAM | Quality |
|--------|-------|------|---------|
| **Standard (PEFT)** | 1x | 16GB | ✓ |
| **Unsloth** | 2-5x | 6-10GB | ✓✓ |

## Installation

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Dry Run (Test)
```bash
python finetune_nemotron_unsloth.py --dry_run
```

### Full Training (Single GPU)
```bash
python finetune_nemotron_unsloth.py \
    --train_file ../datasets/combined_json/train.json \
    --val_file ../datasets/combined_json/val.json \
    --output_dir ../checkpoints/nemotron-unsloth
```

### Multi-GPU Training
```bash
python finetune_nemotron_unsloth.py \
    --gpu_ids "0,1,2" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

## Key Features

### 1. Faster Training
- **2-5x speedup** compared to standard PEFT
- Optimized kernels for attention and FFN
- Efficient gradient checkpointing

### 2. Lower VRAM Usage
- **60% less memory** than standard fine-tuning
- 4-bit quantization optimized
- Can train on RTX 3090/4090 (24GB)

### 3. Automatic Model Merging
```python
# Unsloth automatically saves:
# 1. LoRA adapters only (small)
model.save_pretrained(output_dir)

# 2. Merged model (ready for inference)
model.save_pretrained_merged(
    f"{output_dir}/merged",
    tokenizer,
    save_method="merged_16bit"
)
```

## Inference

### Using Merged Model
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="../checkpoints/nemotron-unsloth/merged",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

# Translate
inputs = tokenizer(
    'Dịch câu sau sang tiếng Việt: "Hello"',
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

### Using Standard Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "../checkpoints/nemotron-unsloth/merged",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("../checkpoints/nemotron-unsloth/merged")
```

## Comparison with Standard Script

| Feature | finetune_nemotron.py | finetune_nemotron_unsloth.py |
|---------|----------------------|------------------------------|
| Speed | 1x | 2-5x |
| VRAM | 16+ GB | 6-10 GB |
| Dependencies | PEFT, bitsandbytes | Unsloth, TRL |
| Model Loading | AutoModelForCausalLM | FastLanguageModel |
| Training | Trainer | SFTTrainer |
| Checkpointing | Standard | Optimized "unsloth" |
| Model Merging | Manual | Automatic |

## Hyperparameters

```bash
python finetune_nemotron_unsloth.py \
    --lora_r 64 \              # LoRA rank
    --lora_alpha 128 \         # LoRA alpha  
    --lora_dropout 0.0 \       # Unsloth recommends 0
    --learning_rate 2e-4 \     # Higher LR works well with Unsloth
    --max_seq_length 512 \     # Context window
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8
```

## Troubleshooting

### Installation Issues
```bash
# If unsloth install fails, try:
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-cache-dir
```

### CUDA Out of Memory
```bash
# Reduce batch size and increase gradient accumulation
python finetune_nemotron_unsloth.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_seq_length 256
```

### Slow Training
```bash
# Disable gradient checkpointing if you have enough VRAM
# (Not recommended, but faster)
# Note: This requires code modification
```

## Tips

1. **Use Unsloth for experimentation** - Faster iteration
2. **Use standard script for production** - More stable
3. **Merge model before deployment** - Easier inference
4. **Monitor VRAM usage** - Unsloth is memory efficient

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [TRL Documentation](https://huggingface.co/docs/trl)
