from pyexpat import model

import torch
import yaml
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zeroshot.zeroshot_evaluator import zeroshot_eval
from models.mgec import MGEC
from utils.logger import Logger


def main():
    config_path = '<path_to_config_yaml>'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    logger = Logger(
        log_name=config['logger']['log_name'],
        log_dir=config['training']['save_dir'],
        log_to_console=config['logger']['log_to_console'],
        log_to_wandb=False,
        log_to_csv=config['logger']['log_to_csv'],
        wandb_config=config['wandb']
    )

    prompt_dict_path = config['zeroshot']['prompt_dict']
    with open(prompt_dict_path, 'r') as f:
        prompt_dict = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MGEC(device=device, network_config=config['network'])
    
    checkpoint_path = '<path_to_checkpoint>'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=True)

    print("Loaded checkpoint successfully.")
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.to(device)
    model.eval()
    
    metrics = {}
    _average_auc, _average_f1, _average_acc, _average_mcc = 0.0, 0.0, 0.0, 0.0

    for set_name in config['zeroshot']['val_sets'].keys():
        res_dict = zeroshot_eval(model, set_name=set_name, device=device, args_zeroshot_eval=config['zeroshot'], logger=logger)

        _average_auc += res_dict['AUROC_AVERAGE']
        _average_f1 += res_dict['F1_AVERAGE']
        _average_acc += res_dict['ACC_AVERAGE']
        _average_mcc += res_dict['MCC_AVERAGE']
        
        for key, value in res_dict.items():
            metrics[f'zeroshot_{set_name}_{key}'] = value

    metrics['zeroshot_val_auc'] = _average_auc / len(config['zeroshot']['val_sets'].keys())
    metrics['zeroshot_val_f1'] = _average_f1 / len(config['zeroshot']['val_sets'].keys())
    metrics['zeroshot_val_accuracy'] = _average_acc / len(config['zeroshot']['val_sets'].keys())
    metrics['zeroshot_val_mcc'] = _average_mcc / len(config['zeroshot']['val_sets'].keys())

    print("Zero-shot evaluation results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

if __name__ == '__main__':
    main()
