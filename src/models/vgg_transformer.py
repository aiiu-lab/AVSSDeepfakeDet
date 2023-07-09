import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


class VGGTransformer(nn.Module):
    def __init__(self, in_dim=1, last_dim=256, frames_per_clip=5, dim_head = 64, depth=3, heads=4, mlp_dim=1024, dropout = 0., emb_dropout = 0.):
        super(VGGTransformer, self).__init__()

        # self.vgg = nn.Sequential(
        #     nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(64, last_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=last_dim),
        #     nn.ReLU(),
        #     nn.Conv2d(last_dim, last_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(num_features=last_dim),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(1, 3),
        # )

        self.vgg = nn.Sequential(
            nn.Conv2d(in_dim, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, last_dim, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1, 3),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, frames_per_clip, last_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(last_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(frames_per_clip * last_dim, frames_per_clip * last_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(frames_per_clip * last_dim // 2, frames_per_clip * last_dim // 4),
        #     nn.ReLU(),
        #     nn.Linear(frames_per_clip * last_dim // 4, frames_per_clip * last_dim // 8)
        # )
        # self.mlp_head = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(last_dim, last_dim),
        #     nn.ReLU(),
        #     nn.Linear(last_dim, last_dim),
        #     nn.ReLU(),
        #     nn.Linear(last_dim, last_dim)
        # )

    def forward(self, x, mask=None):
        b, c, t, h, w = x.shape

        x = rearrange(x, 'b c t h w -> (b t) c h w').contiguous()
        x = self.vgg(x)
        x = rearrange(x, '(b t) ... -> b t ...', b=b, t=t)

        x += self.pos_embedding[:, :t]
        x = self.dropout(x)

        x = self.transformer(x, mask=None)
        x = self.to_latent(x)
        # x = self.mlp_head(x)

        return x