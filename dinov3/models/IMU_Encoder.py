import torch.nn as nn

class IMUEncoder(nn.Module):
    def __init__(self, imu_dim, embed_dim, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.linear = nn.Linear(imu_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, imu):
        """
        imu: [B, T, imu_dim]
        return imu_tokens: [B, num_tokens, embed_dim]
        """
        x = self.linear(imu)  # [B,T,embed_dim]
        x = self.transformer(x)  # [B,T,embed_dim]
        return x[:, : self.num_tokens]
