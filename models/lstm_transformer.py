import torch
import torch.nn as nn


class LSTMTransformerNDVI(nn.Module):
    def __init__(self, input_features=10, d_model=64, hidden_size=128, num_layers=2,
                 num_transformer_layers=3, nhead=4, dim_feedforward=256,
                 dropout=0.1, seq_len=12):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(input_features, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)

        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.lstm_to_transformer = nn.Linear(hidden_size * 2, d_model)
        self.lstm_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_transformer_layers)

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoding.unsqueeze(0)
        lstm_out, _ = self.lstm(x)
        x = self.lstm_norm(self.lstm_to_transformer(lstm_out))
        transformer_out = self.transformer(x)
        x = transformer_out + x
        return self.output_head(x)