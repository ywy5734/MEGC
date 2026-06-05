import argparse
import random
import torch
import numpy as np
import yaml
import logging
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def set_random_seeds(config):
    torch.manual_seed(config['torch_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['torch_seed'])
    np.random.seed(config['numpy_seed'])
    random.seed(config['python_seed'])
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain model.")

    parser.add_argument("--config", type=str, default="<path_to_config_yaml>", help="Path to the configuration file")
    
    parser.add_argument("--train_csv", type=str, help="Path to the training CSV file")
    parser.add_argument("--val_csv", type=str, help="Path to the validation CSV file")
    
    parser.add_argument("--workers", type=int, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")

    parser.add_argument("--criterion_name", type=str, help="Name of the criterion")

    parser.add_argument("--optimizer_name", type=str, help="Name of the criterion")
    parser.add_argument("--optimizer_learning_rate", type=float, help="Learning rate")
    parser.add_argument("--optimizer_weight_decay", type=float, help="Weight decay")
    
    parser.add_argument("--scheduler_name", type=str, help="Name of the scheduler")
    parser.add_argument("--scheduler_mode", type=str, help="Mode of the scheduler")
    parser.add_argument("--scheduler_patience", type=int, help="Patience of the scheduler")
    parser.add_argument("--scheduler_factor", type=float, help="Factor of the scheduler")
    parser.add_argument("--scheduler_verbose", type=bool, help="Verbose of the scheduler")

    parser.add_argument("--scaler_name", type=str, help="Name of the scaler")
    
    parser.add_argument("--model", type=str, help="Model name")
    
    parser.add_argument("--leads", type=int, help="Number of leads in the ECG data")
   
    parser.add_argument("--save_dir", type=str, help="Directory to save the model checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, help="Interval between saved checkpoints")
    
    parser.add_argument("--resume_training", type=bool, help="Resume training")
    parser.add_argument("--resume_from_best", type=bool, help="Resume from best")
    parser.add_argument("--resume_epoch", type=int, help="Resume epoch")
   
    parser.add_argument("--torch_seed", type=int, help="Seed for PyTorch operations")
    parser.add_argument("--numpy_seed", type=int, help="Seed for NumPy random number generators")
    parser.add_argument("--python_seed", type=int, help="Seed for Python random module")
   
    parser.add_argument("--log_name", type=str, help="Name of the logger")
    parser.add_argument("--log_dir", type=str, help="Directory to save the logs")
    parser.add_argument("--log_to_console", type=bool, help="Log to console")
    parser.add_argument("--log_to_wandb", type=bool, help="Log to wandb")
   
    parser.add_argument("--wandb_project", type=str, help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity name")
    parser.add_argument("--wandb_api_key", type=str, help="Wandb API key")
    
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def update_config(config, args):
    def update_dict(d, key, value):
        for k, v in d.items():
            if isinstance(v, dict):
                update_dict(v, key, value)
            if k == key:
                d[k] = value

    for key, value in vars(args).items():
        if value is not None:
            update_dict(config, key, value)
    
    return config

def initialize_training_components(model, criterion_config, optimizer_config, scheduler_config, scaler_config, device, logger):
    # Initialize criterion
    if criterion_config['criterion_name'] == "bce_with_logits_loss":
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        logger.log(f"Unknown criterion: {criterion_config['criterion_name']}", level=logging.ERROR)
    
    # Initialize optimizer
    if optimizer_config['optimizer_name'] == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=optimizer_config['optimizer_learning_rate'], 
            weight_decay=optimizer_config['optimizer_weight_decay'],
            betas=(0.9, 0.999)
        )
    # elif optimizer_config['optimizer_name'] == "adamw":
    #     optimizer = optim.AdamW(
    #         model.parameters(), 
    #         lr=optimizer_config['optimizer_learning_rate'], 
    #         weight_decay=optimizer_config['optimizer_weight_decay'],
    #         betas=(0.9, 0.999)
    #     )

    elif optimizer_config['optimizer_name'] == "adamw":
        ecg_params = []
        text_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "ecg_encoder" in name:
                ecg_params.append(param)
            elif "lm_model" in name:
                text_params.append(param)
            else:
                other_params.append(param)

        if logger:
            logger.log(f"Optimizer param groups:")
            logger.log(f"  ECG encoder params: {len(ecg_params)}")
            logger.log(f"  Text encoder params: {len(text_params)}")
            logger.log(f"  Other params: {len(other_params)}")

        optimizer = optim.AdamW(
            [
                {
                    "params": ecg_params,
                    "lr": optimizer_config.get("ecg_learning_rate", 1e-5),
                    "weight_decay": optimizer_config['optimizer_weight_decay'],
                },
                {
                    "params": text_params,
                    "lr": optimizer_config.get("text_learning_rate", 1e-6),
                    "weight_decay": optimizer_config['optimizer_weight_decay'],
                },
                {
                    "params": other_params,
                    "lr": optimizer_config.get("optimizer_learning_rate", 1e-4),
                    "weight_decay": optimizer_config['optimizer_weight_decay'],
                },
            ],
            betas=(0.9, 0.999)
        )
    else:
        logger.log(f"Unknown optimizer: {optimizer_config['optimizer_name']}", level=logging.ERROR)
    
    # Initialize scheduler
    if scheduler_config['scheduler_name'] == "reduce_lr_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config['scheduler_mode'],
            patience=scheduler_config['scheduler_patience'],
            factor=scheduler_config['scheduler_factor']
        )
    elif scheduler_config['scheduler_name'] == "cosine_annealing_warm_restarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config['scheduler_T_0'],
            T_mult=scheduler_config['scheduler_T_mult'],
            eta_min=scheduler_config['scheduler_eta_min']
        )
    else:
        logger.log(f"Unknown scheduler: {scheduler_config['scheduler_name']}", level=logging.ERROR)

    # Initialize scaler
    if scaler_config['scaler_name'] == "grad_scaler":
        scaler = GradScaler()
    else:
        logger.log(f"Unknown scaler: {scaler_config['scaler_name']}", level=logging.ERROR)

    return criterion, optimizer, scheduler, scaler
