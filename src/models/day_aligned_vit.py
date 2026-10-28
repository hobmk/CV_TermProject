
import torch
import torch.nn as nn


class DayAlignedViT(nn.Module):
    def __init__(
        self,
        image_size=160,
        num_days=20,
        embed_dim=128,
        depth=4,
        num_heads=4,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()

        if image_size % num_days != 0:
            raise ValueError("Image width must be divisible by num_days")

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.image_size = image_size
        self.num_days = num_days
        self.day_width = image_size // num_days

        # Each token covers exactly one trading day:
        # 3 RGB channels x 160 height x 8 width
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(image_size, self.day_width),
            stride=(image_size, self.day_width),
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_days + 1, embed_dim)
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

        if (
            images.ndim != 4
            or tuple(images.shape[1:]) != expected
        ):
            raise ValueError(
                f"Expected [B,3,{self.image_size},{self.image_size}], "
                f"got {tuple(images.shape)}"
            )

        # [B, 3, 160, 160] -> [B, 128, 1, 20]
        x = self.patch_embed(images)

        # [B, 128, 1, 20] -> [B, 20, 128]
        x = x.flatten(2).transpose(1, 2)

        if x.shape[1] != self.num_days:
            raise RuntimeError(
                f"Expected {self.num_days} tokens, got {x.shape[1]}"
            )

        cls = self.cls_token.expand(
            x.size(0), -1, -1
        )

        # [B, 20, 128] -> [B, 21, 128]
        x = torch.cat((cls, x), dim=1)

        x = self.dropout(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x[:, 0])

        return self.head(x)


if __name__ == "__main__":
    model = DayAlignedViT()

    images = torch.randn(2, 3, 160, 160)
    logits = model(images)

    loss = logits.square().mean()
    loss.backward()

    params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Input: {tuple(images.shape)}")
    print(f"Output: {tuple(logits.shape)}")
    print(f"Trading-day tokens: {model.num_days}")
    print(f"Parameters: {params:,}")
    print("[PASS] Forward and backward")
