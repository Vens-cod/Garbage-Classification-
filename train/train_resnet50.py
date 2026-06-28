"""
ResNet50 垃圾分类训练脚本
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image


class GarbageSample:
    def __init__(self, path: str, label: int):
        self.path = path
        self.label = label


class GarbageDataset(Dataset):
    def __init__(self, data_dir: str, list_file: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    rel_path = parts[0]
                    label = int(parts[1])
                    full_path = os.path.join(data_dir, rel_path)
                    self.samples.append(GarbageSample(full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label


def build_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def load_labels(label_list_path: str):
    labels: list[str] = []
    with open(label_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train(args):
    torch.manual_seed(args.seed)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 处理路径
    args.data_dir = os.path.join(base_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    args.train_list = os.path.join(base_dir, args.train_list) if not os.path.isabs(args.train_list) else args.train_list
    args.val_list = os.path.join(base_dir, args.val_list) if not os.path.isabs(args.val_list) else args.val_list
    args.label_list = os.path.join(base_dir, args.label_list) if not os.path.isabs(args.label_list) else args.label_list
    args.out_dir = os.path.join(base_dir, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_ds = GarbageDataset(args.data_dir, args.train_list, transform=build_transform(train=True))
    val_ds = GarbageDataset(args.data_dir, args.val_list, transform=build_transform(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    class_names = load_labels(args.label_list)
    num_classes = len(class_names)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if args.pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    run_name = args.run_name or f"resnet50_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_path = os.path.join(args.out_dir, f"{run_name}.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * x.size(0)
            seen += x.size(0)

        scheduler.step()
        train_loss = running / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "arch": "resnet50",
                "num_classes": num_classes,
                "class_names": class_names,
                "state_dict": model.state_dict(),
                "val_acc": best_acc,
                "epoch": epoch,
            }
            torch.save(ckpt, out_path)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | best={best_acc:.4f}"
        )

    print(f"Saved best checkpoint to: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--train-list", default="data/train_list.txt")
    p.add_argument("--val-list", default="data/validate_list.txt")
    p.add_argument("--label-list", default="data/label_list.txt")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--run-name", default="")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--pretrained", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
