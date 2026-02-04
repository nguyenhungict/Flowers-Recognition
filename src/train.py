import os
import argparse
import yaml
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------- UTILS ----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------- DATA ----------------------
def prepare_dataloaders(root_dir, img_size, batch_size, val_split, num_workers, seed):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform_train)
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
    val_ds.dataset.transform = transform_val

    # In ra thông tin phân bố lớp
    train_counts = Counter([full_dataset.targets[i] for i in train_ds.indices])
    val_counts = Counter([full_dataset.targets[i] for i in val_ds.indices])
    print(f"Train size: {train_size} | Val size: {val_size}")
    print("Train class dist:", {full_dataset.classes[k]: v for k, v in train_counts.items()})
    print("Val class dist:", {full_dataset.classes[k]: v for k, v in val_counts.items()})

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, full_dataset.classes


# ---------------------- MODEL ----------------------
def build_model(model_cfg, num_classes):
    model_type = model_cfg.get("type", "resnet18").lower()
    pretrained = model_cfg.get("pretrained", True)

    if model_type == "resnet18":
        try:
            model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        except:
            model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model


# ---------------------- TRAIN ----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, preds, targets = [], [], []
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(outputs.argmax(1).cpu().numpy())
        targets.extend(labels.cpu().numpy())
    return np.mean(losses), accuracy_score(targets, preds)


def validate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return np.mean(losses), accuracy_score(targets, preds), preds, targets


# ---------------------- PLOTS ----------------------
def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history.png'))
    plt.close()

    # Overfit gap
    plt.figure(figsize=(6, 4))
    plt.plot(np.array(history['train_acc']) - np.array(history['val_acc']))
    plt.title("Overfitting Gap (Train Acc - Val Acc)")
    plt.xlabel("Epoch"); plt.ylabel("Gap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overfitting_gap.png'))
    plt.close()


def save_confusion_matrix(targets, preds, classes, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return cm


# ---------------------- MAIN ----------------------
def main(args):
    cfg = load_config(args.config)
    set_seed(cfg['train'].get('seed', 42))

    device = torch.device('cuda' if (torch.cuda.is_available() and cfg['train']['device'] == 'cuda') else 'cpu')
    print("Using device:", device)

    train_loader, val_loader, classes = prepare_dataloaders(
        cfg['data']['root_dir'], cfg['data']['img_size'],
        cfg['data']['batch_size'], cfg['train']['val_split'],
        cfg['data'].get('num_workers', 4), cfg['train'].get('seed', 42)
    )

    model = build_model(cfg['model'], num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'])

    # ✅ Sửa đoạn scheduler (gọn, không lỗi, không tốn thêm tài nguyên)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(cfg['train']['epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        # ✅ Thêm in ra khi giảm learning rate
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"🔽 Learning rate reduced from {prev_lr:.6f} → {new_lr:.6f}")

        overfit_gap = train_acc - val_acc
        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | Val: loss={val_loss:.4f}, acc={val_acc:.4f} | Overfit gap={overfit_gap:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models/checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classes': classes
            }, "models/checkpoints/best_model_7classes.pth")
            print("✅ Saved new best model.")

    # Final evaluation
    val_loss, val_acc, preds, targets = validate(model, val_loader, criterion, device)
    print(f"\nFinal Val Accuracy: {val_acc:.4f}")

    os.makedirs('outputs/plots', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)

    plot_history(history, 'outputs/plots')
    cm = save_confusion_matrix(targets, preds, classes, 'outputs/plots/confusion_matrix.png')

    # Metrics
    precision = precision_score(targets, preds, average='macro', zero_division=0)
    recall = recall_score(targets, preds, average='macro', zero_division=0)
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    print(f"\nPrecision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}")

    per_class_acc = {classes[i]: cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0 for i in range(len(classes))}
    pd.DataFrame(history).to_csv('outputs/reports/training_log.csv', index=False)

    with open('outputs/reports/training_summary.txt', 'w', encoding='utf-8') as f:
        f.write("===== TRAINING SUMMARY =====\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}\n\n")
        f.write("Per-Class Accuracy:\n")
        for cls, acc in per_class_acc.items():
            f.write(f"- {cls}: {acc*100:.2f}%\n")

    report = classification_report(targets, preds, target_names=classes, digits=4)
    with open('outputs/reports/classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n✅ Training complete! Reports saved in outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args)