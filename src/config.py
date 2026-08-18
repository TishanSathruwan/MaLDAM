"""
Copyright 2026 MaLDAM Authors. All Rights Reserved.

MaLDAM: Masked Localized Domain Adaptation for Malaria Detection in Low-Cost Microscopic Images.
Authors: Muditha Fernando, Tishan Rathnasekara, Saeedha Nazar, Avishka Perera, Tharindu Kaluarachchi
File: Configuration loading & saving
"""

import yaml
from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """
    Loads a configuration from a YAML file.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return OmegaConf.create(config)


def save_config(config: DictConfig, save_path: str) -> None:
    """
    Saves a configuration (DictConfig or dict) to a YAML file.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    with open(save_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
