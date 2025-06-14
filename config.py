# config.py
from types import SimpleNamespace

config = SimpleNamespace(
    image_size=28,
    patch_size=7,
    num_classes=10,
    dim=64,
    depth=6,
    heads=8,
    mlp_dim=128,
    channels=1,
    dropout=0.1,
    emb_dropout=0.1,
    dim_head=32,
    batch_size=64,
    epochs=10,
    lr=3e-4,
    dataset_path='datasets/MNIST',
    device='cpu'  # or 'cpu'
)