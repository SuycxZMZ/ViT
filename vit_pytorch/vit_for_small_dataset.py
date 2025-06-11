import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .vit import ViT, Transformer

# ======================= Helper Function =======================
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ======================= Locality Sensitive Attention (LSA) =======================
class LSA(nn.Module):
    """
    LSA (Locality Sensitive Attention)：
    - 引入 Learnable temperature；
    - 屏蔽对角线（避免注意自身）；
    - 专为小型数据集设计。
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))  # 可学习温度参数

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.heads) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        # Mask: 防止Attention看自身
        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        dots = dots.masked_fill(mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ======================= Small Transformer（继承） =======================
class SmallTransformer(Transformer):
    """
    小型数据集专用 Transformer：
    - 使用LSA替换原Attention。
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                self.layers[0][1]  # 复用FeedForward
            ]) for _ in range(depth)
        ])

# ======================= Shifted Patch Tokenization (SPT) =======================
class SPT(nn.Module):
    """
    Shifted Patch Tokenization (SPT)：
    - 引入上下左右4个偏移版本；
    - 增强局部特征感知；
    - 提升小数据集性能。
    """
    def __init__(self, *, dim, patch_size, channels=3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels  # 原图+4个方向
        self.to_patch_tokens = nn.Sequential(
            rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        # 四方向Shift（padding）
        shifts = [(1, -1, 0, 0),  # 上下
                  (-1, 1, 0, 0),  # 下上
                  (0, 0, 1, -1),  # 左右
                  (0, 0, -1, 1)]  # 右左
        shifted_x = [F.pad(x, shift) for shift in shifts]
        x_with_shifts = torch.cat([x] + shifted_x, dim=1)  # 拼接原图+shift图
        return self.to_patch_tokens(x_with_shifts)

# ======================= SmallViT（继承自ViT） =======================
class SmallViT(ViT):
    """
    专为小数据集设计的ViT（继承自vit.py中的ViT）：
    - 替换PatchEmbedding为SPT；
    - 替换Transformer为SmallTransformer（内含LSA）。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.to_patch_embedding = SPT(dim=kwargs['dim'],
                                      patch_size=kwargs['patch_size'],
                                      channels=kwargs.get('channels', 3))

        self.transformer = SmallTransformer(dim=kwargs['dim'],
                                            depth=kwargs['depth'],
                                            heads=kwargs['heads'],
                                            dim_head=kwargs.get('dim_head', 64),
                                            mlp_dim=kwargs['mlp_dim'],
                                            dropout=kwargs.get('dropout', 0.))