import os
import sys
import argparse
import json
import random
import torch
import numpy as np
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zeroshot.zeroshot_dataset import getdataset
from models.vit1d import vit_base, vit_small, vit_tiny, vit_middle
from utils.logger import Logger


def set_random_seed():
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True



# def set_random_seed(seed=-1):
#     if seed == -1:
#         seed = int(time.time()) + random.randint(0, 10000)

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.benchmark = True 
#     return seed

def parse_args():
    parser = argparse.ArgumentParser(description='ECG Model Fine-Tuning')
    parser.add_argument('--dataset', default='ptbxl_super_class', type=str, help='dataset name')
    parser.add_argument('--ratio', default=100, type=int, help='training data ratio')
    parser.add_argument('--workers', default=4, type=int, help='number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=16, type=int, help='mini_batch size for training')
    parser.add_argument('--test_batch_size', default=256, type=int, help='mini_batch size for testing')
    parser.add_argument('--learning_rate', default=0.003, type=float, help='base learning rate for weights')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--pretrain_path', default='<path_to_pretrained_model>', type=str, help='path to pretrained weights')
    parser.add_argument('--checkpoint_dir', default='<path_to_checkpoint_dir>', type=str, help='path to checkpoint directory')
    parser.add_argument('--backbone', default='vit_tiny', type=str, help='backbone model name')
    parser.add_argument('--num_leads', default=12, type=int, help='number of leads')
    parser.add_argument('--name', default='LinearProbing', type=str, help='experiment name')
    parser.add_argument('--patch_size', default=125, type=int, help='patch size')
    parser.add_argument('--unfreeze_layers', default=3, type=int, help='Number of layers to unfreeze at each step')
    parser.add_argument('--tolerance', default=1000, type=int, help='early stopping tolerance')
    parser.add_argument('--dropout', default=0.4, type=float, help='dropout rate for finetuning')

    parser.add_argument('--seed', default=-1, type=int, help='random seed (-1 for purely random)')

    return parser.parse_args()

def create_model(args, num_classes, logger=None):
    if 'vit' in args.backbone:
        # ---- 修复点 1: 添加 MMLFormer 模型的初始化分支 ----
        if 'mmlformer' in args.backbone:
            from models.mmlformer_vit import mmlformer_vit_tiny, mmlformer_vit_small, mmlformer_vit_middle, mmlformer_vit_base
            if args.backbone == 'mmlformer_vit_tiny':
                # model = mmlformer_vit_tiny(num_classes=num_classes, num_leads=args.num_leads)
                model = mmlformer_vit_tiny(
                    num_classes=num_classes, 
                    num_leads=args.num_leads,
                    use_multilayer_fusion=True,  # 开启多层融合
                    fusion_method='add'       # 使用 concat 或 add
                )
            elif args.backbone == 'mmlformer_vit_small':
                model = mmlformer_vit_small(num_classes=num_classes, num_leads=args.num_leads)
            elif args.backbone == 'mmlformer_vit_middle':
                model = mmlformer_vit_middle(num_classes=num_classes, num_leads=args.num_leads)
            elif args.backbone == 'mmlformer_vit_base':
                model = mmlformer_vit_base(num_classes=num_classes, num_leads=args.num_leads)

        # ----------------------------------------------------
        else:
            if args.backbone == 'vit_tiny':
                model = vit_tiny(num_classes=num_classes, num_leads=args.num_leads, patch_size=args.patch_size)
            elif args.backbone == 'vit_small':
                model = vit_small(num_classes=num_classes, num_leads=args.num_leads, patch_size=args.patch_size)
            elif args.backbone == 'vit_middle':
                model = vit_middle(num_classes=num_classes, num_leads=args.num_leads, patch_size=args.patch_size)
            elif args.backbone == 'vit_base':
                model = vit_base(num_classes=num_classes, num_leads=args.num_leads, patch_size=args.patch_size)
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")
    #     if args.backbone == 'vit_tiny':
    #         model = vit_tiny(num_classes=num_classes, num_leads=args.num_leads, patch_size=args.patch_size)
    #     elif args.backbone == 'vit_small':
    #         model = vit_small(num_classes=num_classes, num_leads=args.num_leads, patch_size=args.patch_size)
    #     elif args.backbone == 'vit_middle':
    #         model = vit_middle(num_classes=num_classes, num_leads=args.num_leads, patch_size=args.patch_size)
    #     elif args.backbone == 'vit_base':
    #         model = vit_base(num_classes=num_classes, num_leads=args.num_leads, patch_size=args.patch_size)

    ckpt = torch.load(args.pretrain_path, map_location='cpu')
    # model.load_state_dict(ckpt['ecg_encoder_state_dict'], strict=False)
    # logger.log(f"Loaded pretrained weights from {args.pretrain_path}")
    if 'ecg_encoder_state_dict' in ckpt:
        state_dict = ckpt['ecg_encoder_state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    clean_state_dict = {}
    for k, v in state_dict.items():
        new_k = k

        if new_k.startswith('module.'):
            new_k = new_k.replace('module.', '', 1)

        if new_k.startswith('backbone.'):
            new_k = new_k.replace('backbone.', '', 1)

        clean_state_dict[new_k] = v

    model_state = model.state_dict()
    matched_state_dict = {}

    for k, v in clean_state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched_state_dict[k] = v

    logger.log(f"Loaded checkpoint from: {args.pretrain_path}")
    logger.log(f"Total checkpoint keys: {len(clean_state_dict)}")
    logger.log(f"Matched keys loaded into model: {len(matched_state_dict)}")
    logger.log(f"First 20 matched keys: {list(matched_state_dict.keys())[:20]}")

    load_msg = model.load_state_dict(matched_state_dict, strict=False)

    logger.log(f"Missing keys after loading: {len(load_msg.missing_keys)}")
    logger.log(f"Unexpected keys after loading: {len(load_msg.unexpected_keys)}")

    if len(matched_state_dict) == 0:
        raise RuntimeError(
            "No pretrained weights were loaded. "
            "Please check whether checkpoint keys match the current model."
        )


    for param in model.parameters():
        param.requires_grad = False
    logger.log(f"Model {args.backbone} backbone frozen.")

    if 'vit' in args.backbone:
        # ---- 修复点 2: 针对 MMLFormer 动态添加 Head，绕过标准 ViT 的硬编码属性 ----
        if 'mmlformer' in args.backbone:
            # 获取特征维度 (MMLFormerViT 中定义为 width)
            in_features = model.width 
            
            # 4.1 安全地包装 MMLFormer 并添加分类头
            # class MMLFormerWrapper(nn.Module):
            #     def __init__(self, backbone, in_features, num_classes):
            #         super().__init__()
            #         self.backbone = backbone
            #         self.head = nn.Linear(in_features, num_classes)
                
            #     def forward(self, x, **kwargs):
            #         # 获取 backbone 输出，pool='mean' 会返回 [B, 1, width]
            #         features = self.backbone(x, pool='mean')
                    
            #         # ！！！修改点：把下面这两行 squeeze 删掉 ！！！
            #         # if features.dim() == 3 and features.size(1) == 1:
            #         #     features = features.squeeze(1)
                    
            #         # 直接传给 head，输出形状变为 [B, 1, num_classes]
            #         # 外面的 output.mean(dim=1) 刚好可以把这个维度为 1 的轴消掉，变成 [B, num_classes]
            #         return self.head(features)
            class MMLFormerWrapper(nn.Module):
                def __init__(self, backbone, in_features, num_classes):
                    super().__init__()
                    self.backbone = backbone
                    self.head = nn.Linear(in_features, num_classes)
                
                def forward(self, x, **kwargs):
                    # 关键修改：提取完全未池化的所有多粒度 Patch 特征
                    # 输出形状为 [B, total_patches, width]
                    features = self.backbone(x, pool='none')
                    
                    # 对所有 Patch 进行全局平均池化，将其压缩为 [B, width]
                    features = features.mean(dim=1)
                    
                    return self.head(features)
            
            model = MMLFormerWrapper(model, in_features, num_classes)
            
            # 2. 解冻新加的分类头
            for param in model.head.parameters():
                param.requires_grad = True
            logger.log(f"Wrapped MMLFormer with Linear Head and unfroze it.")

            # 3. 精准解冻 MMLFormer 的深层网络
            if args.unfreeze_layers > 0:
                # a) 解冻 intra_attention 中的最后 N 个 block
                blocks = model.backbone.intra_attention.blocks
                for block in blocks[-args.unfreeze_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
                
                # b) 必须同时解冻 block 之后的融合层，否则梯度和特征无法完美回传
                for param in model.backbone.aggregation_attn.parameters():
                    param.requires_grad = True
                for param in model.backbone.inter_attention.parameters():
                    param.requires_grad = True
                for param in model.backbone.fusion_norm.parameters():
                    param.requires_grad = True
                    
                logger.log(f"Unfroze the last {args.unfreeze_layers} blocks and subsequent fusion layers of MMLFormer.")
            else:
                logger.log(f"Linear Probing only. MMLFormer backbone is fully frozen.")
        
        # ------------------------------------------------------------------------
        else:
            model.reset_head(num_classes=num_classes)
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True

            # 仅对标准 ViT 解冻指定的 Block
            transformer_blocks = [module for name, module in model._modules.items() if name.startswith('block')]
            for block in transformer_blocks[-args.unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

            for block in range(model.depth):
                transformer_block = getattr(model, f'block{block}')
                if hasattr(transformer_block.attn.fn, 'dropout'):
                    transformer_block.attn.fn.dropout.p = args.dropout
                if hasattr(transformer_block.ff.fn, 'net'):
                    for layer in transformer_block.ff.fn.net:
                        if isinstance(layer, nn.Dropout):
                            layer.p = args.dropout

            logger.log(f"Unfroze the last {args.unfreeze_layers} layers of ViT.")
        # model.reset_head(num_classes=num_classes)
        # model.head.weight.requires_grad = True
        # model.head.bias.requires_grad = True

        # transformer_blocks = [module for name, module in model._modules.items() if name.startswith('block')]
        # for block in transformer_blocks[-args.unfreeze_layers:]:
        #     for param in block.parameters():
        #         param.requires_grad = True

        # for block in range(model.depth):
        #     transformer_block = getattr(model, f'block{block}')
        #     if hasattr(transformer_block.attn.fn, 'dropout'):
        #         transformer_block.attn.fn.dropout.p = args.dropout
        #         print(f"Updated dropout rate for block{block} attention layer.")
        #     if hasattr(transformer_block.ff.fn, 'net'):
        #         for layer in transformer_block.ff.fn.net:
        #             if isinstance(layer, nn.Dropout):
        #                 layer.p = args.dropout
        #                 print(f"Updated dropout rate for block{block} feedforward layer.")

        # logger.log(f"Unfroze the last {args.unfreeze_layers} layers of ViT.")
    
    return model

def load_data(args, data_meta_path, data_split_path):
    if 'ptbxl' in args.dataset:
        dataset_name = 'ptbxl'
        data_path = f'{data_meta_path}/ptbxl'
        data_split_path = os.path.join(data_split_path, f'ptbxl/{args.dataset[6:]}')
    elif args.dataset == 'icbeb':
        dataset_name = args.dataset
        data_path = f'{data_meta_path}/icbeb2018/records500'
        data_split_path = os.path.join(data_split_path, args.dataset)
    elif args.dataset == 'chapman':
        dataset_name = args.dataset
        data_path = f'{data_meta_path}/'
        data_split_path = os.path.join(data_split_path, args.dataset)
    else:
        raise ValueError("Unsupported dataset")

    train_csv_path = os.path.join(data_split_path, f'{args.dataset}_train.csv')
    val_csv_path = os.path.join(data_split_path, f'{args.dataset}_val.csv')
    test_csv_path = os.path.join(data_split_path, f'{args.dataset}_test.csv')

    train_dataset = getdataset(data_path, train_csv_path, mode='train', dataset_name=dataset_name, ratio=args.ratio, backbone=args.backbone)
    val_dataset = getdataset(data_path, val_csv_path, mode='val', dataset_name=dataset_name, backbone=args.backbone)
    test_dataset = getdataset(data_path, test_csv_path, mode='test', dataset_name=dataset_name, backbone=args.backbone)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.num_classes, train_dataset.labels_name

@torch.no_grad()
def evaluate(model, loader, criterion, device='cuda'):
    model.eval()
    y_true, y_pred = [], []
    val_loss = 0

    for ecg, target in loader:
        ecg, target = ecg.to(device), target.to(device)
        output = model(ecg)
        # output = output.mean(dim=1)

        y_true.append(target.cpu().detach().numpy())
        y_pred.append(torch.sigmoid(output).cpu().numpy())

        loss = criterion(output, target)
        val_loss += loss.item()

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    auc = roc_auc_score(y_true, y_pred, average='macro')

    val_loss /= len(loader)
    return auc, val_loss

def main():
    args = parse_args()
    set_random_seed()
    # # 1. 初始化随机种子，并接收当前使用的种子
    # current_seed = set_random_seed(args.seed)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = Logger(
        log_name='ECG Finetuning',
        log_dir=args.checkpoint_dir,
        log_to_console=True,
        log_to_wandb=True,
        wandb_config={
            'wandb_project': 'mgec',
            'wandb_entity': '1',
            'wandb_api_key': '479a'
        }
    )

    wandb.run.name = f"{args.name}_dataset-{args.dataset}_lr-{args.learning_rate}_weightdecay-{args.weight_decay}_epochs-{args.epochs}_batchsize-{args.batch_size}_unfreeze-{args.unfreeze_layers}_ratio-{args.ratio}"
    # wandb.run.save()
    # 3. 将生成的随机种子和其他重要参数打印到本地日志中
    logger.log("========================================")
    logger.log("          Experiment Settings           ")
    logger.log("========================================")
    logger.log(f"Dataset      : {args.dataset}")
    logger.log(f"Backbone     : {args.backbone}")
    logger.log(f"Learning Rate: {args.learning_rate}")
    logger.log(f"Batch Size   : {args.batch_size}")
    logger.log(f"Epochs       : {args.epochs}")
    # logger.log(f"RANDOM SEED  : {current_seed}")  # <--- 核心：在这里把种子记录进日志文件
    logger.log("========================================")
    logger.log("Starting fine-tuning...")



    logger.log("Starting fine-tuning...")

    data_split_path = '<path_to_zeroshot_data_split_dir>'
    data_meta_path = '<path_to_zeroshot_metadata_dir>'
    train_loader, val_loader, test_loader, num_classes, labels_name = load_data(args, data_meta_path, data_split_path)

    model = create_model(args, num_classes, logger)
    model.to(device)    

    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.log(f"{name} is unfrozen and will be trained.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5000,
            T_mult=1,
            eta_min=0.00000001
        )
    scaler = GradScaler()

    train_losses, val_losses = [], []
    best_val_auc = 0
    epochs_no_improve = 0

    for epoch in tqdm(range(args.epochs), desc="Training"):
        model.train()
        train_loss = 0

        for ecg, target in train_loader:
            optimizer.zero_grad()

            # ======== 核心修改点：移除 with autocast() 和 scaler ========
            # 直接使用最稳定的 FP32 精度进行前向传播
            output = model(ecg.to(device))
            # output = output.mean(dim=1)
            loss = criterion(output, target.to(device))
            
            # 直接反向传播
            loss.backward()
            
            # 加入梯度裁剪（给模型上个双保险，防止偶然的突发异常梯度）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 直接更新参数
            optimizer.step()
            # ==========================================================
            # with autocast():
            #     output = model(ecg.to(device))
            #     output = output.mean(dim=1)
            #     loss = criterion(output, target.to(device))
            # scaler.scale(loss).backward()

        


            # scaler.step(optimizer)
            # scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_auc, avg_val_loss = evaluate(model, val_loader, criterion, device)
        test_auc, _ = evaluate(model, test_loader, criterion, device)
        val_losses.append(avg_val_loss)

        logger.log(f'Epoch [{epoch}/{args.epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')
        logger.log_metrics({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss, 'val_auc': val_auc, 'test_auc': test_auc})

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            logger.log(f"Validation AUC improved to {val_auc:.4f}. Model saved.")
        else:
            epochs_no_improve += 1
            logger.log(f"No improvement in Validation AUC for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= args.tolerance:
            logger.log(f"Early stopping triggered. No improvement in Validation AUC for {args.tolerance} consecutive epochs.")
            break

        scheduler.step(avg_val_loss)
        # scheduler.step()

    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(args.checkpoint_dir, 'final_model.pth'))
    # logger.log("Training complete. Model saved.")
    logger.log(f"Training complete. Model saved. | Dataset: {args.dataset} | Data Ratio: {args.ratio}")

if __name__ == '__main__':
    main()
