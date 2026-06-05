import logging
import os
from datetime import datetime
import wandb
import csv
import torch.distributed as dist


class Logger:
    def __init__(self, log_name, log_dir, log_to_console=True, log_to_wandb=False, log_to_csv=True, wandb_config=None):
        self.log_dir = log_dir
        self.log_to_console = log_to_console
        self.log_to_wandb = log_to_wandb
        self.log_to_csv = log_to_csv
        self.wandb_config = wandb_config
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        if self.rank == 0:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        
            self.logger = logging.getLogger(log_name)
            self.logger.setLevel(logging.DEBUG)
        
            # File handler
            log_filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler
            if self.log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                self.logger.addHandler(console_handler)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            if self.log_to_console:
                console_handler.setFormatter(formatter)
        
            self.logger.addHandler(file_handler)
        
            # Initialize wandb
            if self.log_to_wandb and self.wandb_config:
                project = wandb_config['wandb_project']
                entity = wandb_config.get('wandb_entity', None)
                config = wandb_config.get('wandb_config', {})
                wandb.login(key=wandb_config['wandb_api_key'])
                wandb.init(project=project, entity=entity, config=config)

            # Initialize CSV logging if enabled
            if self.log_to_csv:
                self.csv_file = os.path.join(log_dir, "metrics.csv")

        else:
            self.logger = None
    
    def log(self, message, level=logging.INFO):
        if self.rank == 0:
            if level == logging.INFO:
                self.logger.info(message)
            elif level == logging.DEBUG:
                self.logger.debug(message)
            elif level == logging.WARNING:
                self.logger.warning(message)
            elif level == logging.ERROR:
                self.logger.error(message)
            elif level == logging.CRITICAL:
                self.logger.critical(message)
            
            if self.log_to_wandb and self.rank == 0:
                wandb.log({"message": message, "level": level})

    def log_metrics(self, metrics):
        if self.rank == 0:
            if self.log_to_wandb and self.rank == 0:
                wandb.log(metrics)

            if self.log_to_csv:
                self.save_metrics_to_csv(metrics)
        
    def log_params(self, params):
        if self.log_to_wandb and self.rank == 0:
            wandb.config.update(params)

    def save_metrics_to_csv(self, metrics):
        if self.rank == 0:
            file_exists = os.path.isfile(self.csv_file)

            with open(self.csv_file, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics.keys())

                # Write the header only once
                if not file_exists:
                    writer.writeheader()

                writer.writerow(metrics)
