from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
import random
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class MedEVDataCollator:
    tokenizer: PreTrainedTokenizer
    source_lang: str = "vi"  
    target_lang: str = "en"  
    max_source_length: int = 256
    max_target_length: int = 256
    padding: Union[bool, str] = True
    label_pad_token_id: int = -100  
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        
        for feature in features:
            if "translation" in feature:
                sources.append(feature["translation"][self.source_lang])
                targets.append(feature["translation"][self.target_lang])
            elif self.source_lang in feature and self.target_lang in feature:
                sources.append(feature[self.source_lang])
                targets.append(feature[self.target_lang])
            else:
                raise ValueError(
                    f"Feature must contain '{self.source_lang}' and '{self.target_lang}' keys "
                    f"or a 'translation' dict. Got keys: {feature.keys()}"
                )
        

        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )
        

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt",
            )
        
        labels_input_ids = labels["input_ids"]
        labels_input_ids[labels_input_ids == self.tokenizer.pad_token_id] = self.label_pad_token_id
        
        model_inputs["labels"] = labels_input_ids
        
        return model_inputs


@dataclass
class MedEVDataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizer
    model: Optional[Any] = None
    source_lang: str = "vi_VN" 
    target_lang: str = "en_XX"  
    max_source_length: int = 256
    max_target_length: int = 256
    padding: Union[bool, str] = "longest"
    label_pad_token_id: int = -100
    
    def __post_init__(self):
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = self.source_lang
        if hasattr(self.tokenizer, "tgt_lang"):
            self.tokenizer.tgt_lang = self.target_lang
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        
        for feature in features:
            if "translation" in feature:

                src_key = self.source_lang.split("_")[0]
                tgt_key = self.target_lang.split("_")[0]
                sources.append(feature["translation"].get(src_key, feature["translation"].get(self.source_lang)))
                targets.append(feature["translation"].get(tgt_key, feature["translation"].get(self.target_lang)))
            else:
                src_key = self.source_lang.split("_")[0]
                tgt_key = self.target_lang.split("_")[0]
                sources.append(feature.get(src_key, feature.get(self.source_lang)))
                targets.append(feature.get(tgt_key, feature.get(self.target_lang)))
        

        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )
        

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.max_target_length,
                padding=self.padding,
                truncation=True,
                return_tensors="pt",
            )

        labels_ids = labels["input_ids"].clone()
        labels_ids[labels_ids == self.tokenizer.pad_token_id] = self.label_pad_token_id
        model_inputs["labels"] = labels_ids

        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=model_inputs["labels"])
            model_inputs["decoder_input_ids"] = decoder_input_ids
        return model_inputs


@dataclass
class NemotronTranslationCollator:
    """
    Data collator for fine-tuning Nemotron-Nano-9B-v2 as a translation model.
    
    Uses chat template with instruction format for decoder-only LLM SFT.
    Supports bidirectional translation (EN↔VI) with proper label masking.
    
    Format:
        System: /no_think
        User: Translate this English sentence to Vietnamese: '{text}'
        Assistant: {translation}
    """
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
    max_length: int = 512
    padding: Union[bool, str] = True
    label_pad_token_id: int = -100
    bidirectional: bool = True  # Train both EN→VI and VI→EN
    direction: str = "en_to_vi"  # Used if bidirectional=False: "en_to_vi" or "vi_to_en"
    
    # Instruction templates
    en_to_vi_instructions: List[str] = None
    vi_to_en_instructions: List[str] = None
    
    def __post_init__(self):
        # Default instruction templates matching mtet format
        if self.en_to_vi_instructions is None:
            self.en_to_vi_instructions = [
                "Dịch câu sau sang tiếng Việt: \"{text}\"",
            ]
        
        if self.vi_to_en_instructions is None:
            self.vi_to_en_instructions = [
                "Translate the following sentence into English: \"{text}\"",
            ]
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def _create_messages(self, source_text: str, target_text: str, direction: str) -> List[Dict[str, str]]:
        """Create chat messages for translation task."""
        if direction == "en_to_vi":
            instruction = random.choice(self.en_to_vi_instructions).format(text=source_text)
        else:  # vi_to_en
            instruction = random.choice(self.vi_to_en_instructions).format(text=source_text)
        
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": target_text}
        ]
        return messages
    
    def _get_prompt_length(self, source_text: str, direction: str) -> int:
        """Get the token length of just the prompt (without assistant response)."""
        if direction == "en_to_vi":
            instruction = self.en_to_vi_instructions[0].format(text=source_text)
        else:
            instruction = self.vi_to_en_instructions[0].format(text=source_text)
        
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": instruction},
        ]
        
        # Apply chat template without generation prompt to get prompt length
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None
        )
        return len(prompt_tokens)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of translation samples.
        
        Args:
            features: List of dicts with 'en', 'vi', and optionally 'source' keys
            
        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels' tensors
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for feature in features:
            en_text = feature.get("en", "")
            vi_text = feature.get("vi", "")
            
            # Determine translation direction
            if self.bidirectional:
                # Randomly choose direction for each sample
                direction = random.choice(["en_to_vi", "vi_to_en"])
            else:
                direction = self.direction
            
            # Set source and target based on direction
            if direction == "en_to_vi":
                source_text = en_text
                target_text = vi_text
            else:  # vi_to_en
                source_text = vi_text
                target_text = en_text
            
            # Skip empty samples
            if not source_text or not target_text:
                continue
            
            # Create messages
            messages = self._create_messages(source_text, target_text, direction)
            
            # Apply chat template to get full sequence
            full_tokens = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
                truncation=True,
                max_length=self.max_length
            )
            
            # Get prompt length to create labels (mask prompt, keep response)
            prompt_len = self._get_prompt_length(source_text, direction)
            
            # Create labels: -100 for prompt tokens, actual token ids for response
            labels = [self.label_pad_token_id] * prompt_len + full_tokens[prompt_len:]
            
            # Ensure labels length matches input length
            if len(labels) > len(full_tokens):
                labels = labels[:len(full_tokens)]
            elif len(labels) < len(full_tokens):
                labels = labels + [self.label_pad_token_id] * (len(full_tokens) - len(labels))
            
            all_input_ids.append(full_tokens)
            all_attention_masks.append([1] * len(full_tokens))
            all_labels.append(labels)
        
        # Handle empty batch
        if not all_input_ids:
            return {
                "input_ids": torch.tensor([], dtype=torch.long),
                "attention_mask": torch.tensor([], dtype=torch.long),
                "labels": torch.tensor([], dtype=torch.long),
            }
        
        # Pad sequences
        max_len = max(len(ids) for ids in all_input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for input_ids, attention_mask, labels in zip(all_input_ids, all_attention_masks, all_labels):
            pad_len = max_len - len(input_ids)
            
            # Left padding for decoder-only models (common practice)
            padded_input_ids.append([self.tokenizer.pad_token_id] * pad_len + input_ids)
            padded_attention_masks.append([0] * pad_len + attention_mask)
            padded_labels.append([self.label_pad_token_id] * pad_len + labels)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }