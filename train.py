import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from PIL import Image
import time
from model import *

class CXRDataset(Dataset):
    def __init__(self, csv_file, img_dir, num_classes=11, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame.iloc[:]
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        dicom_id = row["Dicom_id"]
        subject_id = row["Subject_id"]
        study_id = row["Study_id"]
        view_pos = row["ViewPosition"]
        patient_ori = row["PatientOrientation"]

        label_ids = [int(x) for x in row["Predict_class"].split(';')]
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_label in label_ids:
            label[cls_label] = 1

        img_path = os.path.join(
            self.img_dir,
            subject_id,
            study_id,
            f"{dicom_id}.jpg"
        )
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

def compute_pos_weight(dataset: CXRDataset, num_classes: int, device: torch.device, clip_max: float = 50.0):
    pos_counts = np.zeros(num_classes, dtype=np.float32)
    df = dataset.data_frame

    for s in df["Predict_class"]:
        for c in [int(x) for x in str(s).split(';')]:
            pos_counts[c] += 1
    total = len(df)
    neg_counts = total - pos_counts

    pos_counts_safe = np.maximum(pos_counts, 1.0)
    pos_weight = neg_counts / pos_counts_safe
    pos_weight = np.clip(pos_weight, 1.0, clip_max)

    return torch.tensor(pos_weight, dtype=torch.float32, device=device)

if __name__ == "__main__":
    start_time = time.time()
    train_dataset = CXRDataset(
        csv_file='csv_files/train.csv',
        img_dir='train',
        num_classes=11,
        transform=train_transform
    )
    valid_dataset = CXRDataset(
        csv_file='csv_files/valid.csv',
        img_dir='valid',
        num_classes=11,
        transform=valid_transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(weights="IMAGENET1K_V1")
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier.in_features, 11)
    )

    pos_weight = compute_pos_weight(train_dataset, num_classes=11, device=device)
    pos_weight[0] = 1.0
    lr = 3e-4
    epochs = 15
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    model.to(device)
    loss_fn.to(device)
    total_train_step = 0
    
    for epoch in range(epochs):
        print(f"---------- 第 {epoch+1} 轮训练开始 ----------")
        epoch_start_time = time.time()
        model.train()
        for imgs, targets in train_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            output = model(imgs)
            loss = loss_fn(output, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_step += 1
            if total_train_step % 50 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss.item()}")

        model.eval()
        probs_all, tgts_all = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs, targets = imgs.to(device), targets.to(device)
                output = model(imgs)
                val_loss_sum += loss_fn(output, targets).item() * imgs.size(0)
                probs_all.append(torch.sigmoid(output).cpu().numpy())
                tgts_all.append(targets.cpu().numpy())

        probs = np.concatenate(probs_all)
        tgts = np.concatenate(tgts_all)

        avg_loss = val_loss_sum / len(valid_dataset)
        valid_cls = tgts.sum(0) > 0
        auc = roc_auc_score(tgts[:, valid_cls], probs[:, valid_cls], average='macro')

        best_f1, best_th = 0, 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            f1 = f1_score(tgts[:, valid_cls], (probs[:, valid_cls] > th).astype(int), average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        print(f"验证集平均 Loss: {avg_loss:.4f}, AUC: {auc:.4f}, F1 Score: {best_f1:.4f}, 最佳阈值TH: {best_th:.2f}")

        torch.save(model.state_dict(), f"mymodel_{epoch}.pth")
        print(f"模型 mymodel_{epoch}.pth 已保存")
        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - epoch_start_time
        print(f"第 {epoch+1} 轮训练耗时: {epoch_elapsed_time:.2f} 秒")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"训练总耗时: {elapsed_time:.2f} 秒")