# ============================================================
# train.py
# FEM-GNN Training Script (Normalized)
# ============================================================

import time
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset import FEMGraphDataset
from model import FEMGNN

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "Data/Simulation_Data/Simulation_Output"
BATCH_SIZE = 2
EPOCHS = 50
LR = 1e-3
SMOOTHNESS_WEIGHT = 0.01
LOG_DIR = "runs/fem_gnn"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Physics-aware smoothness loss
# ----------------------------
def smoothness_loss(pred, edge_index):
    src, dst = edge_index
    diff = pred[src] - pred[dst]
    return (diff ** 2).mean()


def train():
    writer = SummaryWriter(LOG_DIR)

    dataset = FEMGraphDataset(DATA_DIR)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)

    train_set = dataset[:n_train]
    val_set = dataset[n_train:]

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False
    )

    model = FEMGNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse_loss = torch.nn.MSELoss()

    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]",
            leave=False
        )

        for batch in train_pbar:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            pred = model(batch)

            loss_mse = mse_loss(pred, batch.y)
            loss_smooth = smoothness_loss(pred, batch.edge_index)
            loss = loss_mse + SMOOTHNESS_WEIGHT * loss_smooth

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            writer.add_scalar("Batch/Loss", loss.item(), global_step)

            if DEVICE.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1024**2
                train_pbar.set_postfix(
                    loss=f"{loss.item():.2e}",
                    gpu=f"{mem:.0f}MB"
                )
            else:
                train_pbar.set_postfix(loss=f"{loss.item():.2e}")

            global_step += 1

        train_loss /= len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                val_loss += mse_loss(pred, batch.y).item()

        val_loss /= len(val_loader)
        epoch_time = time.time() - start_time

        writer.add_scalar("Epoch/Train_Loss", train_loss, epoch)
        writer.add_scalar("Epoch/Val_Loss", val_loss, epoch)
        writer.add_scalar("Epoch/Time_sec", epoch_time, epoch)

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"Train: {train_loss:.6e} | "
            f"Val: {val_loss:.6e} | "
            f"Time: {epoch_time:.1f}s"
        )

    torch.save(model.state_dict(), "fem_gnn_model.pt")
    writer.close()

    print("\n✅ Training complete")
    print("📦 Model saved as fem_gnn_model.pt")
    print("📊 TensorBoard logs in runs/")

if __name__ == "__main__":
    train()
