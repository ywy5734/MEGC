import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from torch.nn.functional import normalize
from tqdm import tqdm
import numpy as np
import os
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from zeroshot.zeroshot_evaluator import zeroshot_eval
from utils.consistency_loss import ConsistencyLoss
from utils.contrastive_loss import ClipLoss


class Trainer:
    def __init__(
        self,
        model,
        dataset,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        save_path,
        device,
        logger,
        checkpoint_interval=1,
        zeroshot=False,
        zeroshot_configs=None,
        resume_training=False,
        resume_from_best=False,
        resume_epoch=None,
        early_stopping=False,
        early_stopping_patience=None,
        dalr_config=None,
        contrastive_config=None
    ):
        self.model = model
        self.device = device

        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.logger = logger

        self.save_path = save_path
        self.checkpoint_interval = checkpoint_interval
        
        self.device = device

        self.zeroshot = zeroshot
        self.zeroshot_configs = zeroshot_configs

        self.resume_training = resume_training
        self.resume_from_best = resume_from_best
        self.resume_epoch = resume_epoch

        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0

       # --- 初始化 DALR 配置 ---
        self.dalr_config = dalr_config
        self.use_dalr = False
        self.dalr_enable = False
        
        if self.dalr_config and self.dalr_config.get('use_consistency_loss', False):
            self.use_dalr = True
            self.dalr_enable = self.dalr_config.get("enable", False)
            self.dalr_margin = self.dalr_config.get('margin', 0.2)
            self.dalr_weight = self.dalr_config.get('weight', 0.1)
            
            # 初始化损失函数
            # 确保 ConsistencyLoss 类已定义或导入
            self.consistency_criterion = ConsistencyLoss(margin=self.dalr_margin).to(self.device)
            
            self.logger.log(f"DALR Consistency Loss ENABLED. Margin: {self.dalr_margin}, Weight: {self.dalr_weight}")
        else:
            self.logger.log("DALR Consistency Loss DISABLED.")

        
        # --- 新增：初始化 Contrastive (CLIP) 配置 ---
        self.contrastive_config = contrastive_config
        
        self.use_contrastive = False
        
        if self.contrastive_config and self.contrastive_config.get('enable', False):
            self.use_contrastive = True
            self.contrastive_weight = self.contrastive_config.get('weight', 0.5)
            self.contrastive_temp = self.contrastive_config.get('temperature', 0.07)
            
            # 初始化 ClipLoss
            self.clip_criterion = ClipLoss(temperature=self.contrastive_temp).to(self.device)
            
            self.logger.log(f"Contrastive (CLIP) Loss ENABLED. Weight: {self.contrastive_weight}, Temp: {self.contrastive_temp}")
        else:
            self.logger.log("Contrastive (CLIP) Loss DISABLED.")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.train_sampler = train_loader.sampler
        self.val_sampler = val_loader.sampler

    def train_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)

        self.model.train()
        running_loss = 0.0

        self.logger.log(f"Starting training on dataset: {self.dataset.dataset_name}")

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            ecg = batch["ecg"].to(self.device).float()
            labels = batch["labels"].to(self.device).float()  # [B, Num_Classes], 0或1

            self.optimizer.zero_grad()

            with autocast():
                # 根据是否启用 DALR 决定是否返回特征
                # 如果启用 DALR，需要 return_feats=True 来获取特征用于计算相似度矩阵
                if self.dalr_enable:
                    # 获取特征和 logits
                    # ecg_feats: [B, D] (归一化后的全局特征)
                    # all_text_feats: [Num_Classes, D] (所有 Query 的归一化特征)
                    logits, ecg_emb, text_emb = self.model(ecg, self.dataset.pattern_list, return_feats=True)
                else:
                    # 不启用 DALR，只需要 logits
                    logits = self.model(ecg, self.dataset.pattern_list, return_feats=False)
                    ecg_emb, text_emb = None, None
                
                # # 检查 logits 是否包含 NaN 或 Inf
                # if torch.isnan(logits).any() or torch.isinf(logits).any():
                #     nan_count = torch.isnan(logits).sum().item()
                #     inf_count = torch.isinf(logits).sum().item()
                #     self.logger.log(f"WARNING: Found NaN/Inf in logits during training. NaN: {nan_count}, Inf: {inf_count}. Skipping this batch.", level=logging.WARNING)
                #     continue
                
                # 1. 主任务损失 (BCE)
                loss = self.criterion(logits, labels)

        
                
                # # 检查损失是否为 NaN 或 Inf
                # if torch.isnan(loss) or torch.isinf(loss):
                #     self.logger.log(f"WARNING: Loss is NaN/Inf. Skipping this batch.", level=logging.WARNING)
                #     continue

                # 2. 处理特征 (如果是 ViT 输出的序列特征，需要池化)
                ecg_emb_pooled = ecg_emb
                if (self.dalr_enable or self.use_contrastive) and ecg_emb is not None:
                    if ecg_emb.dim() == 3:
                        ecg_emb_pooled = ecg_emb.mean(dim=1)

                # 2. DALR 一致性损失 (Cross-modal Alignment)
                # 注意：MGEC 使用 DistributedDataParallel，如果 ecg_emb 是分散的，
                # 在计算全局对齐时可能需要 gather，但对于单机多卡或仅计算 batch 内对齐，直接计算即可。

                if self.dalr_enable and ecg_emb is not None and text_emb is not None:
                    # 如果 ecg_emb 是 3维 (Batch, Seq_Len, Dim)，说明是 ViT 的序列输出
                    # DALR 需要全局特征，因此我们对序列维度取平均
                    # if ecg_emb.dim() == 3:
                    #     ecg_emb_pooled = ecg_emb.mean(dim=1) 
                    # else:
                    #     ecg_emb_pooled = ecg_emb
                    # 使用池化后的特征计算损失
                    cons_loss = self.consistency_criterion(ecg_emb_pooled, text_emb, labels)
                    
                    # # 检查一致性损失是否为 NaN 或 Inf
                    # if torch.isnan(cons_loss) or torch.isinf(cons_loss):
                    #     self.logger.log(f"WARNING: Consistency loss is NaN/Inf. Skipping consistency loss for this batch.", level=logging.WARNING)
                    # else:
                    #     loss += self.dalr_weight * cons_loss

                    loss += self.dalr_weight * cons_loss


                # 4. (新增) Contrastive (CLIP) 损失
                if self.use_contrastive and ecg_emb is not None:
                    clip_loss = self.clip_criterion(ecg_emb_pooled, text_emb, labels)
                    loss += self.contrastive_weight * clip_loss


            # ---------------------------------------------------------
            # 4. 全局同步的 NaN/Inf 检查 (解决 DDP 死锁与显存泄漏)
            # ---------------------------------------------------------
            is_finite = torch.isfinite(logits).all() and torch.isfinite(loss).all()
            finite_flag = torch.tensor(int(is_finite), device=self.device)

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)

            if finite_flag.item() == 0:
                self.logger.log("WARNING: Found NaN/Inf in logits or loss. Skipping this batch on ALL ranks.", level=logging.WARNING)
                self.optimizer.zero_grad(set_to_none=True)
                
                # 显式删除巨大张量，防止 OOM 显存泄漏
                del logits, loss, finite_flag
                if 'ecg_emb' in locals(): del ecg_emb
                if 'text_emb' in locals(): del text_emb
                if 'ecg_emb_pooled' in locals(): del ecg_emb_pooled
                torch.cuda.empty_cache()
                continue

            # ---------------------------------------------------------
            # 5. 反向传播与标准梯度裁剪 (解决梯度 7000+ 的爆炸问题)
            # ---------------------------------------------------------
            self.scaler.scale(loss).backward()
            
            # AMP 下裁剪梯度前，必须先 unscale
            self.scaler.unscale_(self.optimizer)
            
            # 裁剪梯度，max_norm 设为 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 检查梯度是否为 NaN/Inf
            if not torch.isfinite(grad_norm):
                self.logger.log(f"WARNING: Non-finite grad norm detected ({grad_norm}). Skipping optimizer step.", level=logging.WARNING)
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                
                del logits, loss, grad_norm
                torch.cuda.empty_cache()
                continue

            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # self.scaler.scale(loss).backward()

            # === Debug: 观察 ECG 编码器是否在训练（是否有梯度） ===
            # 只在 rank 0 上、前若干个 batch 打印，避免刷屏
            if (
                dist.get_rank() == 0
                and batch_idx < 5
                and hasattr(self.model, "module")
                and hasattr(self.model.module, "ecg_encoder")
            ):
                with torch.no_grad():
                    total_grad_norm_sq = 0.0
                    grad_exist = False
                    for p in self.model.module.ecg_encoder.parameters():
                        if p.grad is not None:
                            grad_exist = True
                            g = p.grad.detach()
                            total_grad_norm_sq += g.norm(2).item() ** 2
                    if grad_exist:
                        ecg_grad_norm = total_grad_norm_sq ** 0.5
                        self.logger.log(
                            f"[ECG DEBUG] Epoch {epoch} Batch {batch_idx}: ecg_encoder grad L2-norm = {ecg_grad_norm:.6f}"
                        )
                    else:
                        self.logger.log(
                            f"[ECG DEBUG] Epoch {epoch} Batch {batch_idx}: ecg_encoder has NO gradients (all grad=None)."
                        )

            # # 梯度裁剪以防止梯度爆炸
            # self.scaler.unscale_(self.optimizer)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            # 检查损失值是否有效
            loss_value = loss.item()
            if np.isfinite(loss_value):
                running_loss += loss_value * ecg.size(0)
            else:
                self.logger.log(f"WARNING: Loss value is not finite: {loss_value}. Skipping accumulation.", level=logging.WARNING)

        epoch_loss = running_loss / len(self.train_loader.dataset)

        self.logger.log("Training completed.")

        return epoch_loss

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        y_true, y_pred = [], []

        self.logger.log(f"Starting validation on dataset: {self.dataset.dataset_name}")

        metrics = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                ecg = batch["ecg"].to(self.device).float()
                labels = batch["labels"].to(self.device).float()

                outputs = self.model(ecg, self.dataset.pattern_list)
                
                # 兼容性处理：如果是元组则解包，否则直接使用
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # 只取 logits，忽略 ecg_emb 和 text_emb
                else:
                    logits = outputs
                
                # 检查 logits 是否包含 NaN 或 Inf
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    nan_count = torch.isnan(logits).sum().item()
                    inf_count = torch.isinf(logits).sum().item()
                    self.logger.log(f"WARNING: Found NaN/Inf in logits during validation. NaN: {nan_count}, Inf: {inf_count}. Replacing with zeros.", level=logging.WARNING)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                
                loss = self.criterion(logits, labels)
                
                # 检查损失是否有效
                loss_value = loss.item()
                if np.isfinite(loss_value):
                    running_loss += loss_value * ecg.size(0)
                else:
                    self.logger.log(f"WARNING: Loss value is not finite: {loss_value}. Skipping accumulation.", level=logging.WARNING)

                y_true.append(labels.cpu().numpy())

                # Use BCEWithLogitsLoss, manual sigmoid for validation
                # 确保 sigmoid 后的值在有效范围内
                pred_probs = torch.sigmoid(logits)
                pred_probs = torch.clamp(pred_probs, min=0.0, max=1.0)
                y_pred.append(pred_probs.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        # 检查并处理 NaN 值
        nan_mask = np.isnan(y_pred)
        if np.any(nan_mask):
            nan_count = np.sum(nan_mask)
            total_count = y_pred.size
            self.logger.log(f"WARNING: Found {nan_count}/{total_count} NaN values in predictions. Replacing with 0.5.", level=logging.WARNING)
            y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 检查并处理 Inf 值
        inf_mask = np.isinf(y_pred)
        if np.any(inf_mask):
            inf_count = np.sum(inf_mask)
            self.logger.log(f"WARNING: Found {inf_count} Inf values in predictions. Clipping to [0, 1].", level=logging.WARNING)
            y_pred = np.clip(y_pred, 0.0, 1.0)

        # TODO: How to set threshold?
        y_pred_thresholded = (y_pred > 0.5).astype(int)

        auc_scores = []
        for label_idx in range(y_true.shape[1]):
            if len(np.unique(y_true[:, label_idx])) > 1:
                # 检查该标签的预测是否包含 NaN 或 Inf
                label_pred = y_pred[:, label_idx]
                label_true = y_true[:, label_idx]
                
                # 过滤掉 NaN 和 Inf 值（虽然已经处理过，但为了安全起见）
                valid_mask = np.isfinite(label_pred)
                if np.sum(valid_mask) > 0 and len(np.unique(label_true[valid_mask])) > 1:
                    try:
                        auc = roc_auc_score(label_true[valid_mask], label_pred[valid_mask])
                        if np.isfinite(auc):
                            auc_scores.append(auc)
                        else:
                            self.logger.log(f"WARNING: AUC for label {label_idx} is not finite, skipping.", level=logging.WARNING)
                    except ValueError as e:
                        self.logger.log(f"WARNING: Failed to compute AUC for label {label_idx}: {e}, skipping.", level=logging.WARNING)
                else:
                    self.logger.log(f"WARNING: Insufficient valid samples for label {label_idx}, skipping AUC calculation.", level=logging.WARNING)
        average_auc = np.mean(auc_scores) if auc_scores else 0.0

        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
        f1_scores = (
            2 * (recall * precision) / (recall + precision + 1e-10)
        )
        f1 = np.max(f1_scores) if f1_scores.size > 0 else 0
        acc = accuracy_score(y_true, y_pred_thresholded)

        metrics['val_loss'] = epoch_loss
        metrics['val_auc'] = average_auc
        metrics['val_f1'] = f1
        metrics['val_accuracy'] = acc

        self.logger.log("Validation completed.")

        # Zero-shot evaluation
        _average_auc, _average_f1, _average_acc = None, None, None

        if self.zeroshot and self.zeroshot_configs is not None:
            _average_auc, _average_f1, _average_acc = 0.0, 0.0, 0.0

            for set_name in self.zeroshot_configs['val_sets'].keys():
                res_dict = zeroshot_eval(model=self.model.module, set_name=set_name, device=self.device, args_zeroshot_eval=self.zeroshot_configs, logger=self.logger)

                _average_auc += res_dict['AUROC_AVERAGE']
                _average_f1 += res_dict['F1_AVERAGE']
                _average_acc += res_dict['ACC_AVERAGE']

                for key, value in res_dict.items():
                    metrics[f'zeroshot_{set_name}_{key}'] = value
                
            metrics['zeroshot_val_auc'] = _average_auc / len(self.zeroshot_configs['val_sets'].keys())
            metrics['zeroshot_val_f1'] = _average_f1 / len(self.zeroshot_configs['val_sets'].keys())
            metrics['zeroshot_val_accuracy'] = _average_acc / len(self.zeroshot_configs['val_sets'].keys())

            self.logger.log("Zero-shot evaluation completed.")
        
        return metrics

    def train(self, epochs):
        start_epoch = self.resume()

        best_auc_zeroshot = float("-inf") if self.zeroshot else None

        for epoch in range(start_epoch, start_epoch + epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            val_metrics['epoch'] = epoch + 1
            val_metrics['train_loss'] = train_loss
            val_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            self.logger.log_metrics(val_metrics)

            self.logger.log(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_metrics['val_loss']}, Val AUC: {val_metrics['val_auc']}, Val F1: {val_metrics['val_f1']}, Val Accuracy: {val_metrics['val_accuracy']}", level=logging.CRITICAL)
            
            if self.zeroshot:
                self.logger.log(f"Zero-shot AUC: {val_metrics['zeroshot_val_auc']}, Zero-shot F1: {val_metrics['zeroshot_val_f1']}, Zero-shot Accuracy: {val_metrics['zeroshot_val_accuracy']}", level=logging.CRITICAL)

            self.scheduler.step(val_metrics['val_loss'])

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log(f"Current learning rate: {current_lr}", level=logging.CRITICAL)

            if self.zeroshot and val_metrics['zeroshot_val_auc'] > best_auc_zeroshot:
                best_auc_zeroshot = val_metrics['zeroshot_val_auc']
                self.save_model(epoch=epoch+1)

            if epoch % self.checkpoint_interval == 0:
                self.save_model(checkpoint=True, epoch=epoch+1)

            # Early stopping check
            if self.early_stopping:
                if val_metrics['zeroshot_val_auc'] == best_auc_zeroshot:
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    self.logger.log(f"Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}", level=logging.CRITICAL)
                    
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        self.logger.log("Early stopping triggered. Stopping training.")
                        self.save_model(checkpoint=True, epoch=epoch+1)  # Save the model with the correct epoch
                        return

        self.save_model(checkpoint=True, epoch=epochs)

        self.logger.log("All training completed.")

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.ecg_encoder.load_state_dict(checkpoint['ecg_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        self.logger.log(f"Model and optimizer state loaded from {checkpoint_path}")
        
        return start_epoch

    def save_model(self, checkpoint=False, epoch=None):
        if dist.get_rank() == 0:
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'ecg_encoder_state_dict': self.model.module.ecg_encoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler_state_dict': self.scaler.state_dict(),
            }

            if checkpoint and epoch is not None:
                save_path = os.path.join(self.save_path, f"checkpoint_epoch_{epoch}.pth")
            else:
                save_path = os.path.join(self.save_path, "best_model.pth")
            
            torch.save(state, save_path)

            self.logger.log(f"Model state saved to {save_path}")

    def resume(self):
        start_epoch = 0
        
        if not self.resume_training:
            return start_epoch

        if self.resume_from_best:
            best_model_path = os.path.join(self.save_path, "best_model.pth")

            if os.path.exists(best_model_path):
                start_epoch = self.load_model(best_model_path)
                self.logger.log(f"Resuming from best model at epoch {start_epoch}")
            else:
                self.logger.log(f"No best model found at {best_model_path}. Starting new training.")

        elif self.resume_epoch is not None:
            checkpoint_path = os.path.join(self.save_path, f"checkpoint_epoch_{self.resume_epoch}.pth")

            if os.path.exists(checkpoint_path):
                start_epoch = self.load_model(checkpoint_path)
                self.logger.log(f"Resuming from epoch {start_epoch}")
            else:
                self.logger.log(f"No checkpoint found at epoch {self.resume_epoch}. Resuming from the last checkpoint instead.", level=logging.WARNING)
                checkpoint_files = [f for f in os.listdir(self.save_path) if f.startswith("checkpoint_epoch_")]
                if checkpoint_files:
                    last_checkpoint = sorted(checkpoint_files)[-1]
                    start_epoch = self.load_model(os.path.join(self.save_path, last_checkpoint))
                    self.logger.log(f"Resuming from the last checkpoint: {last_checkpoint}")
                else:
                    self.logger.log(f"No checkpoint found. Starting new training.")
        else:
            self.logger.log("Invalid resume configuration. Starting new training.", level=logging.WARNING)

        return start_epoch

    def embedding(self, patterns, model, device):
        tokenizer_output = model._tokenize(patterns).to(device)

        pattern_emb = model.get_text_emb(
            tokenizer_output.input_ids, tokenizer_output.attention_mask
        )

        pattern_emb = model.proj_t(pattern_emb.contiguous())
        pattern_emb = normalize(pattern_emb, dim=-1)

        return pattern_emb
