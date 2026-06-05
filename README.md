# MGEC

本项目是一个基于多模态对齐的大规模心电图（Electrocardiogram, ECG）分类与预训练框架。通过联合临床文本报告与多导联 ECG 信号进行预训练，在多项下游任务中实现了良好的 **Zero-shot** 诊断能力。

## ✨ Key Features

* **MMLFormer：多粒度 ECG 特征编码器**
  采用跨通道多粒度 patching 与两阶段自注意力机制，包括 Intra-Granularity Attention 和 Inter-Granularity Attention，并引入 Router Token 聚合策略。

* **Multi-Layer Fusion：多层级特征融合**
  动态融合 Beginning、Middle 和 Ending 层的 ECG 语义特征，以捕获不同尺度下的心脏病理模式。

* **Dual Local Cross Network：跨模态对齐网络**
  基于 Transformer Query Network (TQN) 和医学语言模型，例如 ClinicalBERT 或 MedCPT，实现 ECG 表征与疾病文本语义之间的跨模态对齐。

* **DALR 与 CLIP-style Contrastive Learning**
  通过一致性损失（DALR）和对比损失（CLIP Loss）增强 ECG 信号与临床文本报告之间的语义一致性。

* **Zero-shot Evaluation**
  支持在主流公开 ECG 数据集上进行零样本评估，包括 PTB-XL、ICBEB 和 Chapman。

* **Distributed Training**
  原生支持 DDP (DistributedDataParallel) 单机多卡训练，并集成 `wandb` 用于实验管理与结果记录。

## 📂 Repository Structure

```text
├── configs/
│   └── mml.yaml                 # Model, training, zero-shot and loss configuration
├── models/
│   ├── dlc.py                   # Dual Local Cross Network (TQN)
│   ├── mgec.py                  # Main multimodal ECG model
│   ├── mmlformer_vit.py         # MMLFormer ViT encoder
│   └── vit1d.py                 # Standard 1D ViT baseline
├── scripts/
│   ├── pretrain.py              # Large-scale ECG-text pre-training script
│   ├── finetune.py              # Fine-tuning and linear probing script
│   └── evaluate.py              # Zero-shot evaluation script
├── utils/
│   ├── consistency_loss.py      # DALR consistency loss
│   ├── contrastive_loss.py      # CLIP-style contrastive loss
│   ├── dataset.py               # MIMIC-IV-ECG pre-training dataset
│   ├── initializer.py           # Optimizer, scheduler and training initialization
│   ├── logger.py                # Console, file and wandb logger
│   └── trainer.py               # Training and validation pipeline
└── zeroshot/
    ├── zeroshot_dataset.py      # Downstream zero-shot datasets
    └── zeroshot_evaluator.py    # Zero-shot metrics: AUROC, F1, ACC and MCC
```

## ⚙️ Installation

建议使用 Conda 管理实验环境。

```bash
conda create -n mmlformer python=3.8
conda activate mmlformer
```

推荐环境：

* Python >= 3.8
* PyTorch >= 2.0
* CUDA >= 11.7
* wandb

安装依赖：

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Pre-training

预训练脚本支持 DDP 多卡并行。模型结构、训练参数以及 DALR/Contrastive Loss 的权重均可通过 `configs/mml.yaml` 进行配置。

```bash
torchrun --nproc_per_node=2 scripts/pretrain.py \
    --config configs/mml.yaml
```

其中，`--nproc_per_node` 应根据实际 GPU 数量进行修改。

### 2. Fine-tuning

可以对预训练得到的 ECG encoder 进行线性探测（Linear Probing）或全量微调（Full Fine-tuning）。同时支持动态解冻指定数量的 Transformer blocks。

```bash
python scripts/finetune.py \
    --dataset ptbxl_super_class \
    --backbone mmlformer_vit_tiny \
    --pretrain_path path/to/your/best_model.pth \
    --checkpoint_dir ./checkpoints/finetune/ \
    --unfreeze_layers 3 \
    --batch_size 64 \
    --learning_rate 0.003
```

若需要进行纯 Linear Probing，可设置：

```bash
--unfreeze_layers 0
```

### 3. Zero-shot Evaluation

可以使用以下命令在 PTB-XL、ICBEB 或 Chapman 数据集上进行零样本评估：

```bash
python scripts/evaluate.py
```

请在 `scripts/evaluate.py` 中修改以下路径：

```python
config_path = "configs/mml.yaml"
checkpoint_path = "path/to/your/best_model.pth"
```

评估指标包括：

* AUROC
* F1-score
* Accuracy
* MCC

## 📝 Configuration

核心配置文件为：

```text
configs/mml.yaml
```

主要配置项包括：

```yaml
network:
  # ECG encoder architecture, e.g., mmlformer_vit_tiny
  # Whether to enable multi-layer fusion
  # Text encoder configuration

training:
  # Batch size
  # Number of epochs
  # Data scale
  # Early stopping

dalr:
  # Whether to enable DALR consistency loss
  # Loss weight
  # Margin parameter

wandb:
  # Project name
  # Run name
  # Logging configuration
```

## 🤝 Contributing

欢迎提交 Issue 或 Pull Request 来完善本框架。

如果您在模型结构、训练配置、数据处理或实验复现过程中遇到问题，可以在 Issue 区留言。
