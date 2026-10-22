import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from model import valid_transform

NUM_CLASSES = 11
CKPT_PATH = "./mymodel_3.pth"
TEST_CSV = "./csv_files/test.csv"
IMG_DIR = "./test"
OUTPUT_CSV = "./test_predictions.csv"
THRESHOLD = 0.55      # ← 强烈建议换成 valid 集选出来的 best_th
SAVE_PROBS = True     # 是否额外输出每一类的概率列


class InferenceDataset(Dataset):
    """只读图像，不需要 Predict_class"""
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(
            self.img_dir,
            str(row["Subject_id"]),
            str(row["Study_id"]),
            f"{row['Dicom_id']}.jpg",
        )
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, idx


def build_model(num_classes=NUM_CLASSES):
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_probs = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs, axis=0)   # (N, NUM_CLASSES)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- 数据 ----------
    dataset = InferenceDataset(TEST_CSV, IMG_DIR, transform=valid_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,          # 必须 False，保证 probs 顺序和 csv 行顺序一致
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    print("Test samples:", len(dataset))

    # ---------- 模型 + 权重 ----------
    model = build_model(NUM_CLASSES)
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print("Loaded weights:", CKPT_PATH)

    # ---------- 推理 ----------
    probs = predict(model, dataloader, device)          # (N, 11)
    preds = (probs > THRESHOLD).astype(int)

    # ---------- 生成 Predict_class ----------
    predict_class_list = []
    for i in range(len(preds)):
        ids = np.where(preds[i] == 1)[0].tolist()
        if not ids:
            # 兜底：全部低于阈值时取概率最高的一类，避免空标签
            ids = [int(np.argmax(probs[i]))]
        predict_class_list.append(";".join(str(x) for x in ids))

    # ---------- 写回 csv（保留原 csv 所有列）----------
    out_df = dataset.df.copy()
    out_df["Predict_class"] = predict_class_list

    if SAVE_PROBS:
        for c in range(NUM_CLASSES):
            out_df[f"prob_class_{c}"] = probs[:, c].round(6)

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to: {OUTPUT_CSV}")