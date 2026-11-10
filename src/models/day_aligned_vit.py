
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
        pooling="cls",
        background_center=False,
        token_norm=False,
        tokenizer="linear",
    ):
        super().__init__()

        if image_size % num_days != 0:
            raise ValueError(
                "image_size must be divisible by num_days"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        if pooling not in ("cls", "mean"):
            raise ValueError(
                "pooling must be 'cls' or 'mean'"
            )

        if tokenizer not in ("linear", "local_cnn"):
            raise ValueError(
                "tokenizer must be 'linear' or 'local_cnn'"
            )

        self.image_size = image_size
        self.num_days = num_days
        self.day_width = image_size // num_days
        self.pooling = pooling
        self.background_center = background_center
        self.token_norm_enabled = token_norm
        self.tokenizer = tokenizer

        # ----------------------------------------------
        # Tokenizer
        # ----------------------------------------------
        if tokenizer == "linear":
            self.patch_embed = nn.Conv2d(
                in_channels=3,
                out_channels=embed_dim,
                kernel_size=(
                    image_size,
                    self.day_width,
                ),
                stride=(
                    image_size,
                    self.day_width,
                ),
            )

        else:
            # Shared CNN for each trading-day slice.
            # Each slice has shape [3, 160, 8].
            self.local_tokenizer = nn.Sequential(
                nn.Conv2d(
                    3, 32, kernel_size=3,
                    stride=(2, 1), padding=1
                ),
                nn.GroupNorm(8, 32),
                nn.GELU(),

                nn.Conv2d(
                    32, 64, kernel_size=3,
                    stride=(2, 2), padding=1
                ),
                nn.GroupNorm(8, 64),
                nn.GELU(),

                nn.Conv2d(
                    64, 64, kernel_size=3,
                    stride=(2, 1), padding=1
                ),
                nn.GroupNorm(8, 64),
                nn.GELU(),

                nn.AdaptiveAvgPool2d((8, 2)),
                nn.Flatten(),
                nn.Linear(
                    64 * 8 * 2, embed_dim
                ),
            )

        self.patch_norm = (
            nn.LayerNorm(embed_dim)
            if token_norm
            else nn.Identity()
        )

        # Retained for old checkpoint compatibility
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_days + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # ----------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------
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
        self.head = nn.Linear(
            embed_dim, num_classes
        )

        nn.init.trunc_normal_(
            self.cls_token, std=0.02
        )

        nn.init.trunc_normal_(
            self.pos_embed, std=0.02
        )

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
                f"Expected [B,3,{self.image_size},"
                f"{self.image_size}], "
                f"got {tuple(images.shape)}"
            )

        if self.background_center:
            images = images - 1.0

        # ----------------------------------------------
        # Trading-Day Tokenization
        # ----------------------------------------------
        if self.tokenizer == "linear":
            # [B,3,160,160] -> [B,D,1,20]
            x = self.patch_embed(images)

            # [B,D,1,20] -> [B,20,D]
            x = x.flatten(2).transpose(1, 2)

        else:
            batch = images.size(0)

            # [B,3,160,160]
            # -> [B,20,3,160,8]
            per_day = (
                images.reshape(
                    batch,
                    3,
                    self.image_size,
                    self.num_days,
                    self.day_width,
                )
                .permute(0, 3, 1, 2, 4)
                .contiguous()
            )

            # [B,20,3,160,8]
            # -> [B*20,3,160,8]
            per_day = per_day.reshape(
                batch * self.num_days,
                3,
                self.image_size,
                self.day_width,
            )

            # Shared CNN
            # [B*20,3,160,8] -> [B*20,D]
            x = self.local_tokenizer(per_day)

            # [B*20,D] -> [B,20,D]
            x = x.reshape(
                batch, self.num_days, -1
            )

        if x.shape[1] != self.num_days:
            raise RuntimeError(
                f"Expected {self.num_days} tokens, "
                f"got {x.shape[1]}"
            )

        x = self.patch_norm(x)

        # ----------------------------------------------
        # Positional Embedding / Pooling
        # ----------------------------------------------
        if self.pooling == "cls":
            cls = self.cls_token.expand(
                x.size(0), -1, -1
            )

            x = torch.cat(
                (cls, x), dim=1
            )

            x = self.dropout(
                x + self.pos_embed
            )

        else:
            x = self.dropout(
                x + self.pos_embed[:, 1:, :]
            )

        for block in self.blocks:
            x = block(x)

        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)

        pooled = self.norm(pooled)

        return self.head(pooled)


if __name__ == "__main__":
    for tokenizer in ("linear", "local_cnn"):
        model = DayAlignedViT(
            tokenizer=tokenizer,
            pooling="mean",
            background_center=True,
            depth=2,
        )

        images = torch.rand(
            2, 3, 160, 160
        )

        logits = model(images)
        logits.square().mean().backward()

        first_layer = (
            model.patch_embed
            if tokenizer == "linear"
            else model.local_tokenizer[0]
        )

        params = sum(
            p.numel() for p in model.parameters()
        )

        print(
            f"[{tokenizer}] "
            f"output={tuple(logits.shape)} "
            f"params={params:,} "
            f"grad_norm="
            f"{first_layer.weight.grad.norm().item():.6f}"
        )
