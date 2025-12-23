# ============================================================
# model.py
# FEM GNN Model
# ============================================================

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class FEMGNN(torch.nn.Module):
    def __init__(self, in_channels=9, hidden=64, out_channels=3):
        super().__init__()

        self.encoder = Linear(in_channels, hidden)

        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

        self.decoder = Linear(hidden, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.encoder(x))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        out = self.decoder(x)

        # ---- HARD BC ENFORCEMENT ----
        # bc_fixed is feature index 3
        bc_mask = data.x[:, 3]
        out = out * (1.0 - bc_mask[:, None])

        return out
