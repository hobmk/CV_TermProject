
import torch
import torch.nn as nn


class StandardViT(nn.Module):
    def __init__(
        self,
        image_size=160,
        patch_size=32,
        embed_dim=128,
        depth=4,
        num_heads=4,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("Image size must be divisible by patch size")

        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by heads")

        self.image_size = image_size
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images):
        expected = (
            3,
            self.image_size,
            self.image_size,
        )

        if images.ndim != 4 or images.shape[1:] != expected:
            raise ValueError(
                f"Expected [B,3,{self.image_size},{self.image_size}], "
                f"got {tuple(images.shape)}"
            )

        x = self.patch_embed(images)

        # [B, D, 5, 5] -> [B, 25, D]
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(
            x.shape[0], -1, -1
        )

        # [B, 25, D] -> [B, 26, D]
        x = torch.cat((cls, x), dim=1)

        x = self.dropout(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])

        return self.head(x)


if __name__ == "__main__":
    model = StandardViT()

    images = torch.randn(2, 3, 160, 160)
    logits = model(images)

    loss = logits.square().mean()
    loss.backward()

    params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Input: {tuple(images.shape)}")
    print(f"Output: {tuple(logits.shape)}")
    print(f"Parameters: {params:,}")
    print("[PASS] Forward and backward")
