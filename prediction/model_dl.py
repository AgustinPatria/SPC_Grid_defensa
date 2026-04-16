"""Paso 2a del Pipeline SPC-Grid: red cuantil (§2.3 del PDF).

f_psi : x_t  -->  ( r_hat^{(q)}_{i, t+1} )_{i in I, q in Q}
con Q = {0.1, 0.3, 0.5, 0.7, 0.9}.

Implementación: LSTM con cabeza de salida por cuantil.
"""
import torch
import torch.nn as nn


QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


class QuantileLSTM(nn.Module):
    """LSTM que predice len(Q) cuantiles para cada activo."""

    def __init__(self, n_assets: int = 2, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.1,
                 quantiles: tuple = QUANTILES):
        super().__init__()
        self.quantiles = quantiles
        self.n_assets = n_assets
        n_q = len(quantiles)

        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_assets * n_q)

    def forward(self, x):
        """
        Args:
            x: (batch, H, n_assets)
        Returns:
            (batch, n_assets, n_quantiles)
        """
        _, (h_n, _) = self.lstm(x)
        out = self.head(h_n[-1])
        return out.view(-1, self.n_assets, len(self.quantiles))
