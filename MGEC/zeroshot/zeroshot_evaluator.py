import torch
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.nn.functional import normalize
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, matthews_corrcoef
import yaml
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zeroshot.zeroshot_dataset import getdataset as get_zero_dataset


def compute_AUCs(gt, pred, n_class):
    AUROCs = []

    gt_np = gt
    pred_np = pred

    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i], average='macro'))

    return AUROCs

def get_class_emd(model, class_name, device='cuda'):
    model.eval()

    with torch.no_grad():
        # Convert class_name list to a single batch
        tokenizer_output = model._tokenize(class_name).to(device)

        # Get text embeddings
        class_embeddings = model.get_text_emb(
            tokenizer_output.input_ids, 
            tokenizer_output.attention_mask
        )

        # Project the embeddings
        class_embeddings = model.proj_t(class_embeddings.contiguous())

        # Normalize the embeddings
        class_embeddings = normalize(class_embeddings, dim=-1)

        # Stack embeddings if necessary
        zeroshot_weights = class_embeddings

    return zeroshot_weights

def get_ecg_emd(model, loader, zeroshot_weights, device='cuda'):
    model.eval()
    y_pred = []
    
    with torch.no_grad():
        for i, (ecg, target) in enumerate(tqdm(loader, desc='Computing Zero-shot AUC')):
            ecg = ecg.to(device=device).float()
            # Predict
            outputs = model(ecg, zeroshot_weights)
                
            # 兼容性处理：如果是元组则解包，否则直接使用
            if isinstance(outputs, tuple):
                logits = outputs[0]  # 只取 logits，忽略 ecg_emb 和 text_emb
            else:
                logits = outputs

            # Apply sigmoid for binary classification
            logits = torch.sigmoid(logits).cpu().numpy()
            
            y_pred.append(logits)
        
    y_pred = np.concatenate(y_pred, axis=0)

    return np.array(y_pred)

def zeroshot_eval(model, set_name, device='cuda', args_zeroshot_eval=None, logger=None):
    assert args_zeroshot_eval is not None, logger.log("Zero-shot evaluation config is None", level=logging.ERROR)

    logger.log(f"Starting zero-shot evaluation on dataset: {set_name}")

    num_workers = args_zeroshot_eval['num_workers']
    batch_size = args_zeroshot_eval['batch_size']
    meta_data_path = args_zeroshot_eval['meta_data_path']

    if 'val_sets' not in args_zeroshot_eval.keys():
        data_path = args_zeroshot_eval['test_sets'][set_name]['data_path']
    if 'val_sets' in args_zeroshot_eval.keys():
        data_path = args_zeroshot_eval['val_sets'][set_name]['data_path']
    
    data_path = os.path.join(meta_data_path, data_path)
    meta_split_path = args_zeroshot_eval['meta_split_path']

    if 'val_sets' not in args_zeroshot_eval.keys():
        split_path = args_zeroshot_eval['test_sets'][set_name]['split_path']
    if 'val_sets' in args_zeroshot_eval.keys():
        split_path = args_zeroshot_eval['val_sets'][set_name]['split_path']
    split_path = os.path.join(meta_split_path, split_path)

    if 'ptbxl' in set_name:
        test_dataset = get_zero_dataset(data_path, split_path, mode='test', dataset_name='ptbxl')
    else:
        test_dataset = get_zero_dataset(data_path, split_path, mode='test', dataset_name=set_name)
    
    class_name = test_dataset.labels_name

    # Open json as dict
    with open(args_zeroshot_eval['prompt_dict'], 'r') as f:
        prompt_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Get prompt for each class
    target_class = [prompt_dict[i] for i in class_name]
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False
    )

    # Get the target array from testset
    gt = test_dataset.labels

    # Get ECG prediction
    pred = get_ecg_emd(model, test_dataloader, target_class, device=device)
    
    AUROCs = compute_AUCs(gt, pred, len(target_class))
    AUROCs = [i * 100 for i in AUROCs]
    AUROC_avg = np.array(AUROCs).mean()
    
    max_f1s = []
    accs = []
    mccs = []

    for i in range(len(target_class)):   
        gt_np = gt[:, i]
        pred_np = pred[:, i]
        precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
        numerator = 2 * recall * precision
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]
        max_f1s.append(max_f1)
        accs.append(accuracy_score(gt_np, pred_np > max_f1_thresh))
        mccs.append(matthews_corrcoef(gt_np, pred_np > max_f1_thresh))

    max_f1s = [i * 100 for i in max_f1s]
    accs = [i * 100 for i in accs]
    f1_avg = np.array(max_f1s).mean()    
    acc_avg = np.array(accs).mean()
    mcc_avg = np.array(mccs).mean()

    res_dict = {
        'AUROC_AVERAGE': AUROC_avg,
        'F1_AVERAGE': f1_avg,
        'ACC_AVERAGE': acc_avg,
        'MCC_AVERAGE': mcc_avg
    }
    for i in range(len(target_class)):
        res_dict.update({
            f'AUROC_{class_name[i]}': AUROCs[i],
            f'F1_{class_name[i]}': max_f1s[i],
            f'ACC_{class_name[i]}': accs[i],
            f'MCC_{class_name[i]}': mccs[i]
        })

    logger.log(f'The average AUROC is {AUROC_avg:.4f}')
    logger.log(f'The average F1 is {f1_avg:.4f}')
    logger.log(f'The average ACC is {acc_avg:.4f}')
    logger.log(f'The average MCC is {mcc_avg:.4f}')

    for i in range(len(target_class)):
        logger.log(f'The AUROC of {class_name[i]} is {AUROCs[i]:.2f}')
        logger.log(f'The F1 of {class_name[i]} is {max_f1s[i]:.2f}')
        logger.log(f'The ACC of {class_name[i]} is {accs[i]:.2f}')
        logger.log(f'The MCC of {class_name[i]} is {mccs[i]:.2f}')

    return res_dict
