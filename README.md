# Hướng Dẫn Chạy Fine-tuning Nemotron cho EN↔VI Translation

## Yêu Cầu Hệ Thống

- **GPU:** NVIDIA GPU với ít nhất 16GB VRAM (A100, RTX 4090, hoặc tương đương)
- **RAM:** 32GB+ khuyến nghị
- **Storage:** ~500GB cho model + dataset
- **CUDA:** 11.8 hoặc mới hơn

---

## Bước 1: Cài Đặt Dependencies

```bash
cd d:\Pythera\pythera

# Cài đặt packages cần thiết
pip install -r requirements.txt

# (Optional) Cài Flash Attention để training nhanh hơn
pip install flash-attn --no-build-isolation
```

---

## Bước 2: Kiểm Tra Dataset

```bash
# Kiểm tra format của dataset
python prepare_dataset.py --check_only

# Kết quả mong đợi:
# Sample 1:
#   EN: Hello, how are you?
#   VI: Xin chào, bạn khỏe không?
#   Source: mtet
```

---

## Bước 3: Dry Run (Test Nhanh)

```bash
# Chạy thử với 10 steps và 100 samples
python finetune_nemotron.py --dry_run

# Kết quả mong đợi:
# - Load model thành công
# - QLoRA adapters được apply
# - Training chạy 10 steps
# - Model được save vào checkpoints/
```

**Nếu gặp lỗi:**
- `CUDA out of memory`: Giảm `per_device_train_batch_size` xuống 1
- `Import error`: Kiểm tra lại requirements.txt
- `Model not found`: Kiểm tra kết nối internet

---

## Bước 4: Training Đầy Đủ

### Option A: Training với Default Settings

```bash
python finetune_nemotron.py \
    --train_file ../datasets/combined_json/train.json \
    --val_file ../datasets/combined_json/val.json \
    --output_dir ../checkpoints/nemotron-translation \
    --num_train_epochs 1
```

### Option B: Training với WandB Logging

```bash
# Đăng nhập WandB (chỉ cần 1 lần)
wandb login

# Training với WandB
python finetune_nemotron.py \
    --train_file ../datasets/combined_json/train.json \
    --val_file ../datasets/combined_json/val.json \
    --output_dir ../checkpoints/nemotron-translation \
    --use_wandb \
    --wandb_project "nemotron-translation" \
    --wandb_run_name "run-1-epoch1" \
    --num_train_epochs 1
```

### Option C: Chọn GPUs Cụ Thể (Multi-GPU)

```bash
# Sử dụng GPU 0, 1, 2 (trong 6 GPUs)
python finetune_nemotron.py \
    --train_file ../datasets/combined_json/train.json \
    --val_file ../datasets/combined_json/val.json \
    --output_dir ../checkpoints/nemotron-translation \
    --gpu_ids "0,1,2" \
    --num_train_epochs 1

# Sử dụng tất cả 6 GPUs
python finetune_nemotron.py \
    --gpu_ids "0,1,2,3,4,5" \
    --use_wandb \
    --num_train_epochs 1
```

### Option D: Training với Custom Settings

```bash
python finetune_nemotron.py \
    --train_file ../datasets/combined_json/train.json \
    --val_file ../datasets/combined_json/val.json \
    --output_dir ../checkpoints/nemotron-translation \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --lora_r 128 \
    --lora_alpha 256 \
    --max_length 768 \
    --save_steps 1000 \
    --eval_steps 1000
```

### Hyperparameters Giải Thích:

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `--lora_r` | 64 | LoRA rank (càng cao càng nhiều params) |
| `--lora_alpha` | 128 | LoRA scaling (thường = 2x rank) |
| `--per_device_train_batch_size` | 2 | Batch size mỗi GPU |
| `--gradient_accumulation_steps` | 8 | Effective batch = 2 × 8 = 16 |
| `--learning_rate` | 2e-5 | Learning rate |
| `--max_length` | 512 | Max sequence length |
| `--num_train_epochs` | 1 | Số epochs |

---

## Bước 5: Monitor Training

### TensorBoard

```bash
# Mở terminal mới
tensorboard --logdir ../checkpoints/nemotron-translation

# Truy cập: http://localhost:6006
```

### Logs

```bash
# Xem logs real-time
tail -f ../checkpoints/nemotron-translation/trainer_state.json
```

---

## Bước 6: Inference (Test Model)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model_path = "d:/Pythera/checkpoints/nemotron-translation"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Test EN → VI
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": 'Dịch câu sau sang tiếng Việt: "Hello, how are you?"'}
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=128,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Expected: "Xin chào, bạn khỏe không?"
```

---

## Troubleshooting

### 1. CUDA Out of Memory

```bash
# Giảm batch size
python finetune_nemotron.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
```

### 2. Training Quá Chậm

```bash
# Tăng batch size nếu có đủ VRAM
python finetune_nemotron.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4
```

### 3. Model Không Học (Loss Không Giảm)

- Tăng learning rate: `--learning_rate 5e-5`
- Tăng LoRA rank: `--lora_r 128 --lora_alpha 256`
- Train nhiều epochs hơn: `--num_train_epochs 3`

### 4. Overfitting

- Giảm learning rate: `--learning_rate 1e-5`
- Tăng weight decay: `--weight_decay 0.1`
- Early stopping: Model tự động save best checkpoint

---

## Thời Gian Ước Tính

| Dataset Size | GPU | Batch Size | Time/Epoch |
|--------------|-----|------------|------------|
| 1.16M samples | A100 (80GB) | 4 | ~12 hours |
| 1.16M samples | A100 (40GB) | 2 | ~24 hours |
| 1.16M samples | RTX 4090 | 1 | ~48 hours |

---

## Evaluation

```bash
# Sau khi training xong, evaluate trên test set
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from datasets import load_dataset
import torch

model = AutoModelForCausalLM.from_pretrained(
    'd:/Pythera/checkpoints/nemotron-translation',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('d:/Pythera/checkpoints/nemotron-translation')

# Load test set và evaluate
# TODO: Implement BLEU score calculation
"
```

---

## Tips

1. **Checkpoint Management:** Model tự động save mỗi 500 steps, giữ 3 checkpoints tốt nhất
2. **Resume Training:** Nếu bị gián đoạn, thêm `--resume_from_checkpoint <path>`
3. **Multi-GPU:** Tự động sử dụng tất cả GPUs với `device_map="auto"`
4. **Memory Optimization:** QLoRA đã optimize memory, không cần thêm gì

---

## Liên Hệ / Issues

Nếu gặp vấn đề, kiểm tra:
1. CUDA version: `nvidia-smi`
2. PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. GPU memory: `nvidia-smi`
