import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import roc_auc_score


# =========================
# Early Stopping
# =========================
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path="best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# =========================
# Model Components
# =========================
class FusionEncoder(nn.Module):
    """
    Multimodal fusion encoder with interaction features and attention.
    """

    def __init__(self, video_dim, audio_dim, hidden_dim):
        super().__init__()

        self.proj_video = nn.Linear(video_dim, hidden_dim)
        self.proj_audio = nn.Linear(audio_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 4,
            num_heads=8,
            batch_first=True
        )

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

    def forward(self, v, a):
        v = self.proj_video(v)
        a = self.proj_audio(a)

        h_mul = v * a
        h_diff = torch.abs(v - a)

        fusion = torch.cat([v, a, h_mul, h_diff], dim=1)
        fusion = fusion.unsqueeze(1)

        z, _ = self.attn(fusion, fusion, fusion)
        z = z.squeeze(1)

        return self.encoder(z)


class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z)).squeeze(-1)


class IMMLModel(nn.Module):
    """
    Full model with causal consistency regularization.
    """

    def __init__(self, video_dim, audio_dim, hidden_dim=256):
        super().__init__()
        self.encoder = FusionEncoder(video_dim, audio_dim, hidden_dim)
        self.classifier = Classifier(hidden_dim)

    def forward(self, v, a, intervention=True):
        z = self.encoder(v, a)
        y = self.classifier(z)

        if not intervention:
            return y

        # Consistency regularization
        loss_cons = 0
        for shift in range(1, 10):
            a_cf = torch.roll(a, shifts=shift, dims=0)
            z_cf = self.encoder(v, a_cf)
            loss_cons += F.l1_loss(z_cf, z.detach())

        return y, loss_cons / 5

    def extract_embedding(self, v, a):
        return self.encoder(v, a)


# =========================
# Data Loader
# =========================
def load_data(video_path, audio_path, label_path):
    video = torch.load(video_path).float()
    audio = torch.load(audio_path).float()

    mat = sio.loadmat(label_path)
    labels = np.mean(mat["all_ratings"], axis=0)
    labels = torch.tensor([1 if x > 0 else 0 for x in labels[:-1]], dtype=torch.float32)

    # alignment for Sherlock
    video = video[26:]
    audio = audio[26:]
    labels = labels[26:946]

    return TensorDataset(video, audio, labels)


# =========================
# Trainer
# =========================
class Trainer:
    def __init__(self, model, device, lr=1e-4, gamma=0.4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for v, a, y in loader:
            v, a, y = v.to(self.device), a.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            y_pred, l_cons = self.model(v, a, True)

            loss = F.binary_cross_entropy(y_pred, y) + self.gamma * l_cons
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        preds, reals = [], []

        with torch.no_grad():
            for v, a, y in loader:
                v, a = v.to(self.device), a.to(self.device)
                p = self.model(v, a, False)
                preds.extend(p.cpu().numpy())
                reals.extend(y.numpy())

        return roc_auc_score(reals, preds)


# =========================
# Embedding Export
# =========================
def export_embeddings(model, dataset, device, save_path):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for v, a, y in loader:
            v, a = v.to(device), a.to(device)
            z = model.extract_embedding(v, a)

            embeddings.append(z.cpu())
            labels.append(y)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    torch.save({"embedding": embeddings, "labels": labels}, save_path)


# =========================
# Main
# =========================
def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    dataset = load_data(
        "/path/video.pt",
        "/path/audio.pt",
        "/path/labels.mat"
    )

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)

    model = IMMLModel(video_dim=768, audio_dim=1024)

    trainer = Trainer(model, device)
    early_stop = EarlyStopping()

    for epoch in range(300):
        loss = trainer.train_epoch(train_loader)
        val_auc = trainer.evaluate(val_loader)

        print(f"Epoch {epoch+1} | Loss {loss:.4f} | Val AUC {val_auc:.4f}")

        early_stop(val_auc, model)
        if early_stop.early_stop:
            break

    # Test
    model.load_state_dict(torch.load("best_model.pt"))
    test_auc = trainer.evaluate(test_loader)
    print(f"Final Test AUC: {test_auc:.4f}")

    # Export embeddings
    export_embeddings(model, dataset, device, "fusion_embeddings.pt")


if __name__ == "__main__":
    main()