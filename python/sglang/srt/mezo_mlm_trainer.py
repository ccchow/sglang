"""
MeZO trainer with MLM (Masked Language Model) support for SGLang.
This implements the MLM trick from the MeZO paper for classification tasks.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass

from sglang.srt.mezo_trainer import MeZOTrainer, MeZODataset
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class MLMConfig:
    """Configuration for MLM-based training."""
    template: str = "It was [MASK]."  # Template to append to text
    label_word_mapping: Dict[int, str] = None  # {0: 'terrible', 1: 'great'}
    use_space_prefix: bool = True  # Add space prefix for RoBERTa/BART/T5


class MeZOMLMTrainer(MeZOTrainer):
    """
    MeZO trainer with MLM objective support.
    
    This implements the clever trick from the MeZO paper where classification
    tasks are converted to language modeling tasks to obtain continuous gradients.
    """
    
    def __init__(
        self, 
        model_runner: ModelRunner, 
        lora_manager: LoRAManager, 
        lora_name: str, 
        tokenizer,
        mlm_config: Optional[MLMConfig] = None,
        **kwargs
    ):
        super().__init__(model_runner, lora_manager, lora_name, tokenizer, **kwargs)
        
        self.mlm_config = mlm_config or MLMConfig()
        self.use_mlm_objective = mlm_config is not None and mlm_config.label_word_mapping is not None
        
        if self.use_mlm_objective:
            # Validate and prepare label word IDs
            self._prepare_label_words()
            logger.info(f"MLM objective enabled with template: {self.mlm_config.template}")
            logger.info(f"Label word mapping: {self.mlm_config.label_word_mapping}")
            logger.info(f"Label word IDs: {self.label_word_ids}")
        else:
            logger.info("Using standard classification objective")
    
    def _prepare_label_words(self):
        """Prepare label word token IDs with proper tokenization."""
        self.label_word_ids = {}
        self.label_words = {}
        
        for label, word in self.mlm_config.label_word_mapping.items():
            # Add space prefix for RoBERTa/BART/T5 if needed
            if self.mlm_config.use_space_prefix and word[0] not in ['<', '[', '.', ',']:
                # Ensure it's a single token
                tokens = self.tokenizer.tokenize(' ' + word)
                if len(tokens) != 1:
                    logger.warning(f"Label word ' {word}' tokenizes to {len(tokens)} tokens: {tokens}")
                token = tokens[0]
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                self.label_words[label] = ' ' + word
            else:
                token_id = self.tokenizer.convert_tokens_to_ids(word)
                self.label_words[label] = word
            
            self.label_word_ids[label] = token_id
    
    def _format_batch_for_mlm(self, batch):
        """Format batch texts with MLM template."""
        if isinstance(batch, dict):
            texts = batch.get('texts', batch.get('prompt', []))
        else:
            texts = [item['prompt'] for item in batch]
        
        # Apply MLM template
        template = self.mlm_config.template
        mlm_texts = []
        
        for text in texts:
            # Simple template replacement
            mlm_text = f"{text} {template}".replace('[MASK]', self.tokenizer.mask_token)
            mlm_texts.append(mlm_text)
        
        return mlm_texts
    
    def _compute_mlm_loss(self, logits, batch):
        """Compute cross-entropy loss on label word vocabulary logits."""
        # Get labels
        if isinstance(batch, dict):
            labels = batch.get('labels', batch.get('label', []))
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
        else:
            labels = [item['completion'] for item in batch]
            # Convert text labels to indices if needed
            if isinstance(labels[0], str):
                # Reverse lookup from word to label
                word_to_label = {v: k for k, v in self.mlm_config.label_word_mapping.items()}
                labels = [word_to_label.get(label, 0) for label in labels]
        
        # Find mask positions in the output
        # This is simplified - in practice, we'd need to track mask positions through tokenization
        batch_size = logits.shape[0]
        device = logits.device
        
        # Extract logits for label words
        label_word_indices = [self.label_word_ids[label] for label in sorted(self.label_word_ids.keys())]
        label_word_indices_tensor = torch.tensor(label_word_indices, device=device)
        
        # Get logits at the last position (simplified - assumes mask is at the end)
        # In a full implementation, we'd track actual mask positions
        mask_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        label_logits = mask_logits[:, label_word_indices_tensor]  # Shape: [batch_size, num_labels]
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        
        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(label_logits, labels_tensor)
        
        return loss.item()
    
    def _forward_pass(self, batch):
        """Perform forward pass with optional MLM formatting."""
        if self.use_mlm_objective:
            # Format texts with MLM template
            mlm_texts = self._format_batch_for_mlm(batch)
            
            # Create requests with MLM texts
            requests = []
            for i, text in enumerate(mlm_texts):
                input_ids = self.tokenizer.encode(text)
                requests.append(Req(
                    rid=str(i),
                    origin_input_text=text,
                    origin_input_ids=input_ids,
                    sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
                    lora_path=self.lora_name,
                ))
            
            # Run forward pass
            schedule_batch = ScheduleBatch.init_new(
                reqs=requests,
                req_to_token_pool=self.model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
                tree_cache=None,
                model_config=self.model_runner.model_config,
                enable_overlap=False,
                spec_algorithm=None,
                enable_custom_logit_processor=False,
            )
            
            schedule_batch.prepare_for_extend()
            model_worker_batch = schedule_batch.get_model_worker_batch()
            
            output, _ = self.model_runner.forward(model_worker_batch)
            
            # Get full logits (not just next token)
            # This requires access to the full vocabulary logits
            # In SGLang's inference mode, this might need modification
            logits = output.next_token_logits
            
            # Compute MLM loss
            loss = self._compute_mlm_loss(logits, batch)
            
            return loss
        else:
            # Use parent class implementation for standard classification
            return super()._forward_pass(batch)


class MLMDataset(MeZODataset):
    """Dataset that formats examples for MLM training."""
    
    def __init__(self, dataset_path, tokenizer, mlm_config: MLMConfig, max_length=512):
        super().__init__(dataset_path, tokenizer, max_length)
        self.mlm_config = mlm_config
    
    def __getitem__(self, idx):
        example = super().__getitem__(idx)
        
        # Add MLM formatting info
        example['mlm_template'] = self.mlm_config.template
        example['label_word'] = self.mlm_config.label_word_mapping.get(
            example.get('label', 0), 
            list(self.mlm_config.label_word_mapping.values())[0]
        )
        
        return example


def create_mlm_config_for_task(task_name: str) -> MLMConfig:
    """Create MLM configuration for common tasks."""
    configs = {
        'sst-2': MLMConfig(
            template="It was [MASK].",
            label_word_mapping={0: 'terrible', 1: 'great'},
            use_space_prefix=True
        ),
        'sst-5': MLMConfig(
            template="It was [MASK].",
            label_word_mapping={
                0: 'terrible',
                1: 'bad',
                2: 'okay',
                3: 'good',
                4: 'great'
            },
            use_space_prefix=True
        ),
        'mnli': MLMConfig(
            template="? [MASK],",
            label_word_mapping={
                0: 'No',      # contradiction
                1: 'Yes',     # entailment
                2: 'Maybe'    # neutral
            },
            use_space_prefix=True
        ),
        'rte': MLMConfig(
            template="? [MASK],",
            label_word_mapping={
                0: 'No',      # not_entailment
                1: 'Yes'      # entailment
            },
            use_space_prefix=True
        ),
    }
    
    return configs.get(task_name.lower())


# Example usage function
def mezo_mlm_finetune(
    model_path: str,
    task_name: str,
    train_dataset: Union[str, List[Dict], MeZODataset],
    lora_rank: int = 8,
    learning_rate: float = 1e-6,
    num_steps: int = 10000,
    epsilon: float = 1e-3,
    batch_size: int = 64,
    max_length: int = 512,
    server_args=None,
    **kwargs
):
    """
    Fine-tune a model using MeZO with MLM objective.
    
    This uses the paper's trick of converting classification to language modeling
    to obtain continuous gradients.
    """
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.hf_transformers_utils import get_tokenizer
    from sglang.srt.configs.model_config import ModelConfig
    
    # Create MLM config for the task
    mlm_config = create_mlm_config_for_task(task_name)
    if mlm_config is None:
        logger.warning(f"No MLM config for task {task_name}, using standard classification")
    
    # Initialize model components (similar to original mezo_finetune)
    if server_args is None:
        server_args = ServerArgs(model_path=model_path, lora_rank=lora_rank)
    
    model_config = ModelConfig(model_path)
    
    # Initialize model runner
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=23333,
        server_args=server_args,
    )
    
    lora_manager = model_runner.lora_manager
    lora_name = "mezo_mlm_lora"
    
    # Create LoRA adapter
    lora_manager.load_lora_adapter(lora_name, "")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(
        model_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code
    )
    
    # Prepare dataset
    if not isinstance(train_dataset, MeZODataset):
        if mlm_config:
            train_dataset = MLMDataset(
                dataset_path=train_dataset,
                tokenizer=tokenizer,
                mlm_config=mlm_config,
                max_length=max_length
            )
        else:
            train_dataset = MeZODataset(
                dataset_path=train_dataset,
                tokenizer=tokenizer,
                max_length=max_length
            )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    # Initialize trainer with MLM support
    trainer = MeZOMLMTrainer(
        model_runner=model_runner,
        lora_manager=lora_manager,
        lora_name=lora_name,
        tokenizer=tokenizer,
        mlm_config=mlm_config,
        normalize_perturbations=False
    )
    
    # Train
    logger.info(f"Starting MeZO training with {'MLM' if mlm_config else 'classification'} objective")
    trainer.train(train_dataloader, learning_rate, num_steps, epsilon)
    
    return {
        'weights': lora_manager.loras[lora_name].weights,
        'config': {
            'model_path': model_path,
            'task_name': task_name,
            'lora_rank': lora_rank,
            'num_steps': num_steps,
            'learning_rate': learning_rate,
            'epsilon': epsilon,
            'use_mlm': mlm_config is not None,
        }
    }