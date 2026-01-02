"""
Fine-tune NVIDIA Nemotron-Nano-9B-v2 for EN↔VI Translation using Unsloth

Unsloth provides 2-5x faster training and 60% less VRAM usage.

Usage:
    python finetune_nemotron.py --dry_run
    python finetune_nemotron.py --gpu_ids "0,1,2"
"""

import os
import sys

# CRITICAL: Set environment variables BEFORE importing anything else
# NOTE: Unsloth may require this to be empty on some setups
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ''
os.environ["HF_DISABLE_TORCHAO"] = "1"

# Optional: Set default GPUs (change as needed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Uncomment to use specific GPUs

# CRITICAL: Import unsloth FIRST before transformers/trl/peft
from unsloth import FastLanguageModel

import json
import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments, set_seed, EarlyStoppingCallback
from trl import SFTTrainer

# Add pythera folder to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_json_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file."""
    logger.info(f"Loading dataset from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples")
    return Dataset.from_list(data)


def formatting_prompts_func(examples, tokenizer):
    """
    Format examples for SFTTrainer using proper chat template.
    
    Each sample is processed TWICE:
    - Once as EN → VI
    - Once as VI → EN
    
    This doubles the effective training data and ensures balanced bidirectional training.
    """
    conversations = []
    
    for i in range(len(examples["en"])):
        en_text = examples["en"][i]
        vi_text = examples["vi"][i]
        
        # Direction 1: EN -> VI
        instruction_en2vi = f'Dịch câu sau sang tiếng Việt: "{en_text}"'
        conversation_en2vi = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": instruction_en2vi},
            {"role": "assistant", "content": vi_text},
        ]
        conversations.append(conversation_en2vi)
        
        # Direction 2: VI -> EN
        instruction_vi2en = f'Translate the following sentence into English: "{vi_text}"'
        conversation_vi2en = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": instruction_vi2en},
            {"role": "assistant", "content": en_text},
        ]
        conversations.append(conversation_vi2en)
    
    # Apply chat template using tokenizer
    texts = [
        tokenizer.apply_chat_template(
            convo, 
            tokenize=False, 
            add_generation_prompt=False
        ) 
        for convo in conversations
    ]
    
    return {"text": texts}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron with Unsloth")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    
    # Data arguments
    parser.add_argument("--train_file", type=str, default="../dataset/train.json")
    parser.add_argument("--val_file", type=str, default="../dataset/val.json")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)  # Unsloth recommends 0
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="../checkpoints/nemotron-translation")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", default=False)
    # Precision: use --bf16 for bf16 or --no-fp16 to disable fp16
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)  # Default bf16 for RTX 5090
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="nemotron-translation")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # GPU selection
    parser.add_argument("--gpu_ids", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set GPU devices
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logger.info(f"Using GPUs: {args.gpu_ids}")
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize WandB
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
    
    # Dry run settings
    if args.dry_run:
        args.max_steps = 10
        args.max_train_samples = 100
        args.max_val_samples = 20
        args.logging_steps = 1
        args.save_steps = 10
        args.eval_steps = 10
        logger.info("DRY RUN mode enabled")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("NEMOTRON TRANSLATION FINE-TUNING (UNSLOTH)")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA rank: {args.lora_r}")
    logger.info(f"Output: {args.output_dir}")
    
    # =========================================================================
    # Load model with Unsloth
    # =========================================================================
    logger.info("\n[1/5] Loading model with Unsloth...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        #dtype=torch.float16,   # fp16 + QLoRA
        load_in_4bit=False,
        load_in_8bit=False,  # Disabled - causes shape mismatch with Mamba layers
        trust_remote_code=True,
        device_map="auto", 
        unsloth_force_compile = True,
        attn_implementation="eager",  
    )
    logger.info("=== DEBUG MODEL INFO ===")
    logger.info(f"Model dtype: {model.dtype}")
    logger.info(f"Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
    logger.info(f"Is 4-bit loaded (QLoRA)? {getattr(model, 'is_loaded_in_4bit', False)}")
    logger.info(f"BitsAndBytes config: {getattr(model, 'bnb_config', None)}")
    logger.info(f"Model modules using LoRA:")
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            logger.info(f"  {name} -> LoRA detected")
    logger.info("=========================")
    
    logger.info(f"✓ Model loaded with Unsloth (2-5x faster training)")
    logger.info(f"  Max seq length: {args.max_seq_length}")
    
    # =========================================================================
    # Apply LoRA
    # =========================================================================
    logger.info("\n[2/5] Applying LoRA with Unsloth optimizations...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",      # Attention layers
            "gate_proj", "up_proj", "down_proj",          # MLP layers
            "in_proj", "out_proj",                         # Mamba layers (CRITICAL for Nemotron!)
        ],
        bias="none",
        use_gradient_checkpointing = "unsloth",  # Unsloth's optimized checkpointing
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )
    
    logger.info(f"✓ LoRA applied (rank={args.lora_r}, alpha={args.lora_alpha})")
    
    # =========================================================================
    # Load datasets
    # =========================================================================
    logger.info("\n[3/5] Loading datasets...")
    
    train_dataset = load_json_dataset(args.train_file)
    if args.max_train_samples:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    logger.info(f"  Train: {len(train_dataset):,} samples")
    
    val_dataset = load_json_dataset(args.val_file)
    if args.max_val_samples:
        val_dataset = val_dataset.select(range(min(args.max_val_samples, len(val_dataset))))
    logger.info(f"  Val: {len(val_dataset):,} samples")
    
    # Format datasets using chat template
    logger.info("  Formatting datasets for SFT...")
    
    # Create a wrapper function that passes tokenizer
    def format_with_tokenizer(examples):
        return formatting_prompts_func(examples, tokenizer)
    
    train_dataset = train_dataset.map(
        format_with_tokenizer,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        format_with_tokenizer,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    logger.info("✓ Datasets formatted with chat template")
    
    # =========================================================================
    # Training arguments
    # =========================================================================
    logger.info("\n[4/6] Configuring trainer...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
       # fp16=args.fp16,
        #bf16=args.bf16,
        optim="adamw_8bit",  # 8-bit AdamW for memory efficiency
        weight_decay=0.01,
        lr_scheduler_type="linear",
        max_steps=args.max_steps,
        seed=args.seed,
        report_to=["tensorboard", "wandb"] if args.use_wandb else "none",  # Use 'none' to avoid JSON serialization issues
        run_name=args.wandb_run_name if args.use_wandb else None,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # =========================================================================
    # Create SFTTrainer
    # =========================================================================
    # Early stopping callback - stops training if eval_loss doesn't improve for 3 evals
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Stop after 3 evals without improvement
        early_stopping_threshold=0.01,  # Minimum improvement threshold
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        packing=False,  # Can set True for more efficiency with short sequences
        callbacks=[early_stopping_callback],
    )
    
    logger.info("✓ SFTTrainer configured")
    
    # =========================================================================
    # Apply response-only training (OPTIONAL - requires correct format detection)
    # =========================================================================
    # NOTE: Temporarily disabled - need to detect correct Nemotron chat template format
    # The model will train on full sequences (both instruction and response)
    # This is still valid for fine-tuning, just slightly less efficient
    
    # To enable, uncomment below and set correct instruction_part/response_part:
    # from unsloth.chat_templates import train_on_responses_only
    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part="<user_token>",  # TODO: Detect actual token
    #     response_part="<assistant_token>",
    # )
    
    logger.info("\n[5/6] Skipping train_on_responses_only (training on full sequence)")
    
    # =========================================================================
    # Train
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    # Enable Unsloth's faster training mode
    FastLanguageModel.for_training(model)
    
    train_result = trainer.train()
    
    # =========================================================================
    # Save model
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING MODEL")
    logger.info("=" * 60)
    
    # Save LoRA adapters
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"✓ LoRA adapters saved to: {args.output_dir}")
    
    # Save merged model for easier inference
    logger.info("  Merging and saving full model...")
    try:
        model.save_pretrained_merged(
            f"{args.output_dir}/merged",
            tokenizer,
            save_method="merged_16bit",  # Can use "merged_4bit" for smaller size
        )
        logger.info(f"✓ Merged model saved to: {args.output_dir}/merged")
    except Exception as e:
        logger.warning(f"Could not save merged model: {e}")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Final loss: {metrics.get('train_loss', 'N/A')}")
    logger.info(f"Models saved to: {args.output_dir}")
    
    # =========================================================================
    # Quick inference test
    # =========================================================================
    if not args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("TESTING INFERENCE")
        logger.info("=" * 60)
        
        FastLanguageModel.for_inference(model)  # Enable inference mode (faster)
        
        test_input = 'Dịch câu sau sang tiếng Việt: "Hello, how are you?"'
        
        inputs = tokenizer(
            f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

/no_think<|eot_id|><|start_header_id|>user<|end_header_id|>

{test_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nTest translation:")
        logger.info(f"  Input: {test_input}")
        logger.info(f"  Output: {response}")


if __name__ == "__main__":
    main()
