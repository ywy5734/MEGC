import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms
import logging


class MIMIC_E_T_Dataset(Dataset):
    def __init__(self, train_npy_path, val_npy_path, transform=None, logger=None, **args):
        self.mode = args["train_test"]
        self.pattern_list = args["pattern_list"]
        self.logger = logger

        if self.mode == "train":
            self.ecg_data = np.load(train_npy_path, mmap_mode="r")
        else:
            self.ecg_data = np.load(val_npy_path, mmap_mode="r")

        self.text_csv = args["text_csv"]

        assert (
            self.ecg_data.shape[0] == self.text_csv.shape[0]
        ), self.logger.log("Data size mismatch!", level=logging.ERROR)

        self.transform = transform

    def __len__(self):
        return self.text_csv.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # we have to divide 1000 to get the real value
        ecg = self.ecg_data[idx] / 1000

        # get raw text
        report = self.text_csv.iloc[idx]["total_report"]

        # get pseudo labels
        labels = self.text_csv.loc[idx, self.pattern_list].values.tolist()
        labels = torch.tensor(labels, dtype=torch.long)

        sample = {"ecg": ecg, "raw_text": report, "labels": labels}

        if self.transform:
            if self.mode == "train":
                sample["ecg"] = self.transform(sample["ecg"])
                sample["ecg"] = torch.squeeze(sample["ecg"], dim=0)
            else:
                sample["ecg"] = self.transform(sample["ecg"])
                sample["ecg"] = torch.squeeze(sample["ecg"], dim=0)

        return sample


class ECG_TEXT_Dataset:
    def __init__(self, dataset_name, train_csv_path, val_csv_path, train_npy_path, val_npy_path, logger):
        self.dataset_name = dataset_name
        self.logger = logger

        self.logger.log("Loading MIMIC dataset...")
        
        self.train_csv = pd.read_csv(train_csv_path, low_memory=False)
        self.val_csv = pd.read_csv(val_csv_path, low_memory=False)

        self.train_npy_path = train_npy_path
        self.val_npy_path = val_npy_path

        self.pattern_list = self.train_csv.columns[3:].tolist()

        self.logger.log(f"Loaded {len(self.pattern_list)} patterns, train size: {self.train_csv.shape[0]}, val size: {self.val_csv.shape[0]}")

    def get_dataset(self, train_test, T=None):
        if train_test == "train":
            self.logger.log("Applying Train-stage Transform...")

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        else:
            self.logger.log("Applying Val-stage Transform...")

            Transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

        if self.dataset_name == "mimic":
            if train_test == "train":
                misc_args = {
                    "train_test": train_test,
                    "text_csv": self.train_csv,
                    "pattern_list": self.pattern_list,
                }
            else:
                misc_args = {
                    "train_test": train_test,
                    "text_csv": self.val_csv,
                    "pattern_list": self.pattern_list,
                }

            dataset = MIMIC_E_T_Dataset(self.train_npy_path, self.val_npy_path, Transforms, self.logger, **misc_args)
            self.logger.log(f"Loaded {train_test} dataset")

        return dataset
