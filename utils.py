# utils.py
import os
import torch

def accuracy(preds, labels):
    _, pred_classes = torch.max(preds, dim=1)
    return (pred_classes == labels).float().mean().item()

def save_checkpoint(model, optimizer, epoch, acc, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc
    }, filename)

def load_checkpoint(model, filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from {filename}, epoch {checkpoint['epoch']}, acc {checkpoint['accuracy']:.4f}")