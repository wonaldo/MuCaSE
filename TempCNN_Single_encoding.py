import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.input_data import NiftiLabelsMasker


# =========================
# Config
# =========================
class Config:
    FMRI_ROOT = "./fmri"
    EMB_PATH = "./data/fusion_embeddings.pt"
    SAVE_PATH = "./results/sherlock_cnn_results.csv"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    HIDDEN_DIM = 256
    LR = 1e-4
    EPOCHS = 300
    PATIENCE = 10
    DROP_TR = 26
    HRFLAG = 2

    N_PERM = 1000
    P_THRESHOLD = 0.05


# =========================
# Model: 3-layer CNN
# =========================
class TemporalCNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=9, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)


# =========================
# Loss
# =========================
class FMRILoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        l2 = self.mse(pred, target)
        cos = 1 - nn.functional.cosine_similarity(pred, target, dim=-1).mean()
        return (1 - self.alpha) * l2 + self.alpha * cos


# =========================
# EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore(self, model):
        model.load_state_dict(self.best_state)


# =========================
# Permutation Test
# =========================
def block_permutation_test(y_true, y_pred, n_perm=1000, block_size=10):
    """
    Block-wise permutation test to preserve temporal structure
    """

    def split_blocks(x):
        n_blocks = len(x) // block_size
        x = x[:n_blocks * block_size]
        return x.reshape(n_blocks, block_size)

    # real correlation
    real_r, _ = pearsonr(y_true, y_pred)

    # reshape into blocks
    y_blocks = split_blocks(y_true)
    pred_blocks = split_blocks(y_pred)

    n_blocks = y_blocks.shape[0]

    perm_rs = []

    for _ in range(n_perm):
        perm_idx = np.random.permutation(n_blocks)

        y_perm = y_blocks[perm_idx].reshape(-1)
        pred_perm = pred_blocks[perm_idx].reshape(-1)

        r, _ = pearsonr(y_perm, pred_perm)
        perm_rs.append(r)

    perm_rs = np.array(perm_rs)

    p_val = np.mean(perm_rs >= real_r)

    return real_r, p_val


# =========================
# Main
# =========================
def main():
    cfg = Config()

    os.makedirs("./results", exist_ok=True)

    atlas = fetch_atlas_schaefer_2018(n_rois=100)
    atlas_path = atlas["maps"]
    roi_names = atlas["labels"][1:]

    masker = NiftiLabelsMasker(labels_img=atlas_path, strategy="mean")

    z_all = torch.load(cfg.EMB_PATH)["fusion_embedding"].numpy()

    fmri_files = sorted(glob.glob(os.path.join(cfg.FMRI_ROOT, "sub-*", "func", "*bold.nii.gz")))

    subject_data = {}

    for f in fmri_files:
        sid = os.path.basename(f).split("_")[0]

        img = nib.load(f)
        ts = masker.fit_transform(img)

        ts = ts[cfg.DROP_TR:]
        z_cut = z_all[cfg.DROP_TR:]

        z_aligned = z_cut[:-cfg.HRFLAG]
        y_aligned = ts[cfg.LAG:]

        subject_data[sid] = (z_aligned, y_aligned)

    subjects = list(subject_data.keys())
    n_roi = len(roi_names)

    results = []

    for test_sid in subjects:
        print(f"Test: {test_sid}")

        z_train, y_train = [], []

        for sid in subjects:
            if sid == test_sid:
                continue
            z_i, y_i = subject_data[sid]
            z_train.append(z_i)
            y_train.append(y_i)

        z_train = np.concatenate(z_train)
        y_train = np.concatenate(y_train)

        z_test, y_test = subject_data[test_sid]

        z_mean, z_std = z_train.mean(0), z_train.std(0) + 1e-8
        y_mean, y_std = y_train.mean(0), y_train.std(0) + 1e-8

        z_train = (z_train - z_mean) / z_std
        z_test = (z_test - z_mean) / z_std
        y_train = (y_train - y_mean) / y_std
        y_test_norm = (y_test - y_mean) / y_std

        z_train_t = torch.tensor(z_train, dtype=torch.float32).to(cfg.DEVICE)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(cfg.DEVICE)
        z_test_t = torch.tensor(z_test, dtype=torch.float32).to(cfg.DEVICE)

        model = TemporalCNN(z_train.shape[-1], cfg.HIDDEN_DIM, n_roi).to(cfg.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
        loss_fn = FMRILoss()

        early_stop = EarlyStopping(cfg.PATIENCE)

        split = int(0.9 * len(z_train))
        z_tr, z_val = z_train_t[:split], z_train_t[split:]
        y_tr, y_val = y_train_t[:split], y_train_t[split:]

        for _ in range(cfg.EPOCHS):
            model.train()
            optimizer.zero_grad()
            pred = model(z_tr.unsqueeze(0)).squeeze(0)
            loss = loss_fn(pred, y_tr)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(z_val.unsqueeze(0)).squeeze(0)
                val_loss = loss_fn(val_pred, y_val).item()

            if early_stop.step(val_loss, model):
                break

        early_stop.restore(model)

        model.eval()
        with torch.no_grad():
            test_pred = model(z_test_t.unsqueeze(0)).squeeze(0).cpu().numpy()

        r_list = []

        for i in range(n_roi):
            r, p = block_permutation_test(
                y_test_norm[:, i],
                test_pred[:, i],
                cfg.N_PERM
            )

            if p <= cfg.P_THRESHOLD:
                r_list.append(r)
            else:
                r_list.append(0)

        results.append(r_list)

    df = pd.DataFrame(results, index=subjects, columns=roi_names)
    df.to_csv(cfg.SAVE_PATH)

    print("Done")
    print("Mean r:", df.mean().mean())


if __name__ == "__main__":
    main()