# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Union, List, Optional

@dataclass
class AttackConfig:
    """Attack configuration"""
    attack_types: Union[str, List[str]]
    attack_modes: Union[str, List[str]]
    p: float = 0.5  # poison ratio

@dataclass
class TriggerConfig:
    """Trigger configuration"""
    epsilon: float
    mask_size: int
    input_dim: int
    hidden_dim: int
    trigger_model: str
    trigger_weight: Optional[str] = None
    img_dim: int = 3

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    manual_classes: Optional[str]
    generate_upper_bound: int
    bias: float
    modify_image: str
    stage: str
    top_n: int

@dataclass
class TrainingConfig:
    """Training configuration"""
    lr: float
    save_interval: int
    by_epoch: bool = True
    save_begin: int = 0
    save_last: bool = True

@dataclass
class ValidationConfig:
    """Validation configuration"""
    metrics: List[str]
    noise_test: bool = False
    fp16: bool = False
