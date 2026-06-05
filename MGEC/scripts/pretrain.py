import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.dataset import ECG_TEXT_Dataset
from models.mgec import MGEC
from utils.trainer import Trainer
from utils.logger import Logger
from utils.initializer import set_random_seeds, parse_args, load_config, update_config, initialize_training_components


def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    torch.cuda.empty_cache()

    # Set the GPU device for the current process
    torch.cuda.set_device(dist.get_rank())
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    
    # Parse arguments
    args = parse_args()

    # Load configs
    config = load_config(args.config)
    config = update_config(config, args)
    
    # Initialize logger
    logger = Logger(
        log_name=config['logger']['log_name'],
        log_dir=config['training']['save_dir'],
        log_to_console=config['logger']['log_to_console'],
        log_to_wandb=config['logger']['log_to_wandb'],
        log_to_csv=config['logger']['log_to_csv'],
        wandb_config=config['wandb']
    )
    
    # Log initial parameters
    logger.log_params(config)
    logger.log(f"Using device: {device}, Current device: {torch.cuda.current_device()}, Total devices: {torch.cuda.device_count()}")
    logger.log("Starting pretraining process")

    # Set random seed for reproducibility
    set_random_seeds(config['seed'])
    logger.log(f"Random seeds set as {config['seed']}")

    # Initialize dataset and dataloaders
    dataset = ECG_TEXT_Dataset(
        dataset_name=config['training']['dataset'],
        train_csv_path=config['training']['train_csv'],
        val_csv_path=config['training']['val_csv'],
        train_npy_path=config['training']['train_npy'],
        val_npy_path=config['training']['val_npy'],
        logger=logger,
        
    )

    train_dataset = dataset.get_dataset("train")
    val_dataset = dataset.get_dataset("val")

    # Apply data scaling
    data_scale = config['training'].get('data_scale', 1.0)  # Default to 1.0 (100% of data)
    if data_scale < 1.0:
        train_size = len(train_dataset)
        subset_size = int(len(train_dataset) * data_scale)
        indices = random.sample(range(len(train_dataset)), subset_size)
        train_dataset = Subset(train_dataset, indices)
        logger.log(f"Using {subset_size}/{train_size} samples for training (scale: {data_scale})")

    # Use DistributedSampler for train and validation datasets
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'], 
        drop_last=True, 
        shuffle=False, 
        num_workers=config['training']['workers'],
        sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['training']['batch_size'], 
        drop_last=True, 
        shuffle=False, 
        num_workers=config['training']['workers'],
        sampler=val_sampler
    )

    # Initialize model
    config['network']['num_classes'] = len(dataset.pattern_list)
    model = MGEC(
        device,
        network_config=config['network']
    )
    model = model.to(device)

    # Check lm_encoder parameters before DDP
    for name, param in model.lm_model.named_parameters():
        logger.log(f"Layer: {name} | Requires Grad: {param.requires_grad}")

    # Wrap model with DDP
    model = DistributedDataParallel(model, device_ids=[dist.get_rank()], find_unused_parameters=True)

    # Initialize criterion, optimizer, scheduler and scaler
    criterion, optimizer, scheduler, scaler = initialize_training_components(
        model=model, 
        criterion_config=config['criterion'], 
        optimizer_config=config['optimizer'], 
        scheduler_config=config['scheduler'],
        scaler_config=config['scaler'],
        device=device,
        logger=logger
    )

    # Initialize Trainer
    
    trainer = Trainer(
        model,
        dataset,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        config['training']['save_dir'],
        device,
        logger,
        config['training']['checkpoint_interval'],
        config['training']['zeroshot'],
        config['zeroshot'],
        config['training']['resume_training'],
        config['training']['resume_from_best'],
        config['training']['resume_epoch'],
        config['training']['early_stopping'],
        config['training']['early_stopping_patience'],
        # --- 新增：传递 DALR 配置 ---
        dalr_config=config.get('dalr', None), # 使用 get 防止配置文件中没写报错

        contrastive_config=config.get('contrastive', None)
        
    
    )
    trainer.train(config['training']['epochs'])

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
