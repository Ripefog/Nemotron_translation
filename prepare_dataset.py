"""
Prepare and preprocess datasets for Nemotron translation fine-tuning.

This script converts JSON datasets to HuggingFace Dataset format
and optionally creates tokenized versions for faster loading.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --output_format arrow
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_json_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)


def analyze_dataset(dataset: Dataset, name: str = "Dataset"):
    """Print dataset statistics."""
    logger.info(f"\n{name} Statistics:")
    logger.info(f"  - Total samples: {len(dataset):,}")
    
    # Count sources
    sources = {}
    en_lengths = []
    vi_lengths = []
    
    for sample in dataset:
        source = sample.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
        
        en_lengths.append(len(sample.get("en", "").split()))
        vi_lengths.append(len(sample.get("vi", "").split()))
    
    logger.info(f"  - Sources: {sources}")
    logger.info(f"  - Avg EN words: {sum(en_lengths)/len(en_lengths):.1f}")
    logger.info(f"  - Avg VI words: {sum(vi_lengths)/len(vi_lengths):.1f}")
    logger.info(f"  - Max EN words: {max(en_lengths)}")
    logger.info(f"  - Max VI words: {max(vi_lengths)}")


def prepare_datasets(
    train_file: str,
    val_file: str,
    test_file: str,
    output_dir: str,
    output_format: str = "arrow",
    max_samples: Optional[int] = None,
):
    """Prepare and save datasets."""
    
    logger.info("=" * 60)
    logger.info("PREPARING DATASETS")
    logger.info("=" * 60)
    
    # Load datasets
    logger.info("\n[1/4] Loading datasets...")
    
    train_dataset = load_json_dataset(train_file)
    val_dataset = load_json_dataset(val_file)
    test_dataset = load_json_dataset(test_file)
    
    if max_samples:
        train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(max_samples // 10, len(val_dataset))))
        test_dataset = test_dataset.select(range(min(max_samples // 10, len(test_dataset))))
    
    # Analyze
    logger.info("\n[2/4] Analyzing datasets...")
    analyze_dataset(train_dataset, "Train")
    analyze_dataset(val_dataset, "Validation")
    analyze_dataset(test_dataset, "Test")
    
    # Create DatasetDict
    logger.info("\n[3/4] Creating DatasetDict...")
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })
    
    # Save
    logger.info(f"\n[4/4] Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    if output_format == "arrow":
        dataset_dict.save_to_disk(output_dir)
        logger.info(f"Saved as Arrow format to: {output_dir}")
    else:
        # Save as JSON
        for split_name, split_data in dataset_dict.items():
            output_path = os.path.join(output_dir, f"{split_name}.json")
            split_data.to_json(output_path)
            logger.info(f"Saved {split_name} to: {output_path}")
    
    logger.info("\nDataset preparation complete!")
    
    return dataset_dict


def check_sample_format(json_file: str, num_samples: int = 3):
    """Check format of samples in JSON file."""
    logger.info(f"\nChecking sample format from {json_file}:")
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for i, sample in enumerate(data[:num_samples]):
        logger.info(f"\nSample {i+1}:")
        logger.info(f"  EN: {sample.get('en', 'N/A')[:100]}...")
        logger.info(f"  VI: {sample.get('vi', 'N/A')[:100]}...")
        logger.info(f"  Source: {sample.get('source', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="d:/Pythera/datasets/combined_json/train.json"
    )
    parser.add_argument(
        "--val_file", 
        type=str, 
        default="d:/Pythera/datasets/combined_json/val.json"
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        default="d:/Pythera/datasets/combined_json/test.json"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="d:/Pythera/datasets/processed"
    )
    parser.add_argument(
        "--output_format", 
        type=str, 
        choices=["arrow", "json"],
        default="arrow"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Max samples for debugging"
    )
    parser.add_argument(
        "--check_only", 
        action="store_true",
        help="Only check sample format, don't process"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        check_sample_format(args.train_file)
        return
    
    prepare_datasets(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        output_dir=args.output_dir,
        output_format=args.output_format,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
