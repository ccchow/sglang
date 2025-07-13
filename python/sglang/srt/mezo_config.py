"""
MeZO Configuration with default hyperparameters from the paper.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class MeZOConfig:
    """Configuration for MeZO training following the paper's hyperparameters."""
    
    # Core MeZO hyperparameters (from paper Table 15)
    learning_rate: float = 1e-6  # Paper grid: {1e-7, 1e-6, 1e-5}
    epsilon: float = 1e-3  # Zero-order perturbation scale
    batch_size: int = 64  # Paper default for RoBERTa
    num_steps: int = 100_000  # Paper default
    eval_steps: int = 10_000  # Evaluate every 1/10 of total steps
    
    # Optimizer settings
    optimizer: str = "sgd"  # Paper uses SGD with MeZO
    weight_decay: float = 0.0  # Paper default
    momentum: float = 0.0  # No momentum in basic MeZO
    lr_scheduler: str = "constant"  # Paper uses constant LR
    
    # MeZO-specific settings
    normalize_perturbations: bool = False  # Paper doesn't normalize
    efficient_zero_order: bool = True  # Memory-efficient implementation
    
    # LoRA settings (when using MeZO + LoRA)
    lora_r: int = 8  # Paper default
    lora_alpha: int = 16  # Paper default
    lora_dropout: float = 0.0
    
    # Advanced options
    clip_grad_norm: Optional[float] = None  # No gradient clipping by default
    seed: int = 42
    
    # Task-specific overrides
    task_configs: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Apply task-specific configurations if provided."""
        if self.task_configs is None:
            self.task_configs = self._get_default_task_configs()
    
    def _get_default_task_configs(self) -> Dict[str, Dict[str, Any]]:
        """Default configurations for specific tasks from the paper."""
        return {
            "sst-2": {
                "learning_rate_grid": [1e-7, 1e-6, 1e-5],
                "batch_size": 64,
                "num_steps": 100_000,
            },
            "mnli": {
                "learning_rate_grid": [1e-7, 1e-6, 1e-5],
                "batch_size": 64,
                "num_steps": 100_000,
            },
            "squad": {
                "learning_rate_grid": [1e-7, 5e-7, 1e-6],
                "batch_size": 16,
                "num_steps": 20_000,
            },
            # When using prefix tuning
            "prefix": {
                "learning_rate_grid": [1e-3, 5e-3, 1e-2],
                "epsilon": 1e-1,
                "num_prefix_tokens": 5,
            },
            # When using LoRA
            "lora": {
                "learning_rate_grid": [1e-5, 5e-5, 1e-4],
                "epsilon": 1e-3,
                "weight_decay": 0.1,
            }
        }
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get task-specific configuration."""
        task_name = task_name.lower()
        if task_name in self.task_configs:
            return self.task_configs[task_name]
        return {}
    
    def update_for_task(self, task_name: str):
        """Update configuration for a specific task."""
        task_config = self.get_task_config(task_name)
        for key, value in task_config.items():
            if hasattr(self, key) and key != "learning_rate_grid":
                setattr(self, key, value)


def get_mezo_config_for_model(model_name: str, task_name: str = None) -> MeZOConfig:
    """Get MeZO configuration based on model and task."""
    config = MeZOConfig()
    
    # Model-specific adjustments
    if "roberta-large" in model_name.lower():
        config.batch_size = 64
    elif "opt-13b" in model_name.lower():
        config.batch_size = 16
        config.num_steps = 20_000  # Shorter for larger models
    elif "opt-30b" in model_name.lower() or "opt-66b" in model_name.lower():
        config.batch_size = 8
        config.num_steps = 10_000
        config.eval_steps = 0  # No intermediate evaluation for very large models
    
    # Task-specific adjustments
    if task_name:
        config.update_for_task(task_name)
    
    return config