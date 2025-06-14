# train.py
import os
import torch
from tqdm import tqdm
from torch import nn, optim
from config import config
from utils import accuracy, save_checkpoint, load_checkpoint
from dataset import get_dataloaders
from vit_pytorch.vit import ViT
from vit_pytorch.vit_for_small_dataset import SmallViT

def get_model():
    model = ViT(
        image_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.num_classes,
        dim=config.dim, 
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        channels=config.channels,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout
    ).to(config.device)
    return model

def train():
    train_loader, val_loader, _ = get_dataloaders()
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0

    for epoch in range(config.epochs):
        model.train()
        total_loss, total_acc = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.epochs}]")

        for imgs, labels in pbar:
            imgs, labels = imgs.to(config.device), labels.to(config.device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            acc = accuracy(output, labels)
            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix(loss=loss.item(), acc=acc)

        avg_acc = total_acc / len(train_loader)
        print(f"[Train] Epoch {epoch+1} | Loss: {total_loss:.4f} | Acc: {avg_acc:.4f}")

        # 保存最近一次
        save_checkpoint(model, optimizer, epoch + 1, avg_acc, filename="checkpoints/last_checkpoint.pth")

        # 验证并更新最佳模型
        val_acc = validate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch + 1, val_acc, filename="checkpoints/best_checkpoint.pth")
            print(f"✅ Best model updated: {val_acc:.4f}")

def validate(model=None, loader=None):
    _, val_loader, _ = get_dataloaders() if loader is None else (None, loader, None)
    if model is None:
        model = get_model()
        load_checkpoint(model, filename="checkpoints/best_checkpoint.pth")

    model.eval()
    total_acc = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(config.device), labels.to(config.device)
            output = model(imgs)
            acc = accuracy(output, labels)
            total_acc += acc
    avg_acc = total_acc / len(val_loader)
    print(f"[Val] Accuracy: {avg_acc:.4f}")
    return avg_acc
def test():
    _, _, test_loader = get_dataloaders()
    model = get_model()
    load_checkpoint(model, filename="checkpoints/best_checkpoint.pth")

    model.eval()
    total_acc = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(config.device), labels.to(config.device)
            output = model(imgs)
            acc = accuracy(output, labels)
            total_acc += acc
    avg_acc = total_acc / len(test_loader)
    print(f"[Test] Accuracy: {avg_acc:.4f}")