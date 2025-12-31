"""
Fine-tune NVIDIA Nemotron-Nano-9B-v2 for EN↔VI Translation using QLoRA

Usage:
    python finetune_nemotron.py --config config.yaml
    python finetune_nemotron.py --train_file train.json --val_file val.json --output_dir ./checkpoints
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Add pythera folder to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data_collator import NemotronTranslationCollator

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name: str = field(
        default="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        metadata={"help": "Pretrained model name or path"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention 2 if available"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code from HuggingFace"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    train_file: str = field(
        default="d:/Pythera/datasets/combined_json/train.json",
        metadata={"help": "Path to training data JSON file"}
    )
    val_file: str = field(
        default="d:/Pythera/datasets/combined_json/val.json",
        metadata={"help": "Path to validation data JSON file"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max training samples (for debugging)"}
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max validation samples (for debugging)"}
    )
    bidirectional: bool = field(
        default=True,
        metadata={"help": "Train both EN→VI and VI→EN directions"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA/QLoRA configuration."""
    use_lora: bool = field(
        default=True,
        metadata={"help": "Use LoRA for efficient fine-tuning"}
    )
    use_qlora: bool = field(
        default=True,
        metadata={"help": "Use QLoRA (4-bit quantization + LoRA)"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha (usually 2x rank)"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout rate"}
    )
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        metadata={"help": "Modules to apply LoRA"}
    )


def load_json_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file."""
    logger.info(f"Loading dataset from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples")
    return Dataset.from_list(data)


def get_quantization_config(use_qlora: bool) -> Optional[BitsAndBytesConfig]:
    """Get quantization config for QLoRA."""
    if not use_qlora:
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(args: LoraArguments) -> LoraConfig:
    """Get LoRA configuration."""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron for translation")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="d:/Pythera/datasets/combined_json/train.json")
    parser.add_argument("--val_file", type=str, default="d:/Pythera/datasets/combined_json/val.json")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--use_qlora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="d:/Pythera/checkpoints/nemotron-translation")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Run a quick test with 10 steps")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", default=True,
                        help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="nemotron-translation",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    
    # GPU selection
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g., '0,1,2')")
    
    args = parser.parse_args()
    
    # Set GPU devices if specified
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info(f"Using GPUs: {args.gpu_ids}")
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize WandB if enabled
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            logger.info(f"WandB initialized: project={args.wandb_project}")
        except ImportError:
            logger.warning("WandB not installed. Install with: pip install wandb")
            args.use_wandb = False
    
    # Override for dry run
    if args.dry_run:
        args.max_steps = 10
        args.max_train_samples = 100
        args.max_val_samples = 20
        args.logging_steps = 1
        args.save_steps = 10
        args.eval_steps = 10
        logger.info("DRY RUN mode enabled - running 10 steps with small dataset")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("NEMOTRON TRANSLATION FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"QLoRA: {args.use_qlora}, LoRA rank: {args.lora_r}")
    logger.info(f"Output: {args.output_dir}")
    
    # =========================================================================
    # Load tokenizer
    # =========================================================================
    logger.info("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set padding side for decoder-only models
    tokenizer.padding_side = "left"
    
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # =========================================================================
    # Load datasets
    # =========================================================================
    logger.info("\n[2/5] Loading datasets...")
    
    train_dataset = load_json_dataset(args.train_file)
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    
    val_dataset = load_json_dataset(args.val_file)
    if args.max_val_samples:
        val_dataset = val_dataset.select(range(min(args.max_val_samples, len(val_dataset))))
    logger.info(f"Validation dataset: {len(val_dataset)} samples")
    
    # =========================================================================
    # Load model with quantization
    # =========================================================================
    logger.info("\n[3/5] Loading model...")
    
    quantization_config = get_quantization_config(args.use_qlora)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not args.use_qlora else None,
        trust_remote_code=args.trust_remote_code,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")
    
    # =========================================================================
    # Apply LoRA
    # =========================================================================
    if args.use_lora:
        logger.info("\n[4/5] Applying LoRA...")
        
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        logger.info("\n[4/5] Full fine-tuning (no LoRA)")
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # =========================================================================
    # Create data collator
    # =========================================================================
    logger.info("\n[5/5] Creating data collator...")
    
    data_collator = NemotronTranslationCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        bidirectional=args.bidirectional,
    )
    
    # =========================================================================
    # Training arguments
    # =========================================================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard", "wandb"] if args.use_wandb else ["tensorboard"],
        run_name=args.wandb_run_name if args.use_wandb else None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=args.max_steps,
        seed=args.seed,
        optim="paged_adamw_8bit" if args.use_qlora else "adamw_torch",
    )
    
    # =========================================================================
    # Create Trainer
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # =========================================================================
    # Train
    # =========================================================================
    train_result = trainer.train()
    
    # =========================================================================
    # Save model
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info(f"\nModel saved to: {args.output_dir}")
    logger.info("Training complete!")
    
    # =========================================================================
    # Quick inference test
    # =========================================================================
    if not args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("TESTING INFERENCE")
        logger.info("=" * 60)
        
        test_messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "Translate this English sentence to Vietnamese: 'Hello, how are you?'"}
        ]
        
        inputs = tokenizer.apply_chat_template(
            test_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test translation:\n{response}")


if __name__ == "__main__":
    main()
