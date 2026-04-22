#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import pearsonr
from tqdm.auto import tqdm

from ridge_utils.ridge import bootstrap_ridge
# from ridgeutils.ridge import bootstrap_ridge

# =========================
# Relative Paths and Fixed Config
# =========================

# Base directory: location of this script
CODE_ROOT = Path(__file__).parent.resolve()
RESULT_ROOT = CODE_ROOT / "result"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CH = 100
TOTAL_SEEDS = 140
TEST_SEEDS = [0, 20, 40, 60, 80, 100, 120]
RIDGE_FULL_SEEDS = list(range(120, 140))
RIDGE_TEST_SEEDS = [120]

# Atlas files expected under ./data/atlas/
ATLAS_NII_PATH = str(CODE_ROOT / "data" / "atlas" / "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
ATLAS_TXT_PATH = str(CODE_ROOT / "data" / "atlas" / "Schaefer2018_100Parcels_7Networks_order.txt")

ALPHAS = np.logspace(1, 3, 10)
NBOOTS = 1
CHUNKLEN = 40
NCHUNKS = 20
DELAYS = range(0, 6)  # 0, 1, 2, 3, 4, 5

# Dataset configurations (using relative paths)
DATASETS = {
    "fnl": {
        "feature_path": str(CODE_ROOT / "data" / "features" / "fnl_embeddings.pt"),
        "expected_feature_tr": 1358,
        "fmri_mode": "fnl",
        "fmri_dir": str(CODE_ROOT / "data" / "fnl" / "derivatives" / "denoised" / "smoothed"),
        "trim_start": 5,
        "trim_end": 1,
        "expected_target_tr": 1358,
    },
    "sherlock": {
        "feature_path": str(CODE_ROOT / "data" / "features" / "sherlock_embeddings.pt"),
        "expected_feature_tr": 920,
        "fmri_mode": "sherlock",
        "fmri_root": str(CODE_ROOT / "data" / "sherlock" / "fmriprep"),
        "trim_start": 26,
        "trim_end": 0,
        "expected_target_tr": 920,
    },
}

TRAIN_CONFIG = {
    "hidden_dim": 128,
    "kernel_size": 9,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "max_epochs": 200,
    "patience": 15,
}

BASE_METHOD_ORDER = [
    "tcn2_mse",
    "tcn3_mse",
    "tcn4_mse",
    "tcn2_pearson",
    "tcn3_pearson",
    "tcn4_pearson",
    "bootstrap_ridge",
]

ALL_METHOD_ORDER = [
    "tcn2_mse",
    "tcn3_mse",
    "tcn4_mse",
    "tcn2_pearson",
    "tcn3_pearson",
    "tcn4_pearson",
    "bootstrap_ridge",
    "all_model_mean",
    "all_tcn_mean",
]

METHOD_TO_FILENAME = {
    "tcn2_mse": "01_tcn2_mse.csv",
    "tcn3_mse": "02_tcn3_mse.csv",
    "tcn4_mse": "03_tcn4_mse.csv",
    "tcn2_pearson": "04_tcn2_pearson.csv",
    "tcn3_pearson": "05_tcn3_pearson.csv",
    "tcn4_pearson": "06_tcn4_pearson.csv",
    "bootstrap_ridge": "07_bootstrap_ridge.csv",
    "all_model_mean": "08_all_model_mean.csv",
    "all_tcn_mean": "09_all_tcn_mean.csv",
}


# =========================
# Random Seed
# =========================

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================
# Custom Loss
# =========================

class PearsonCorrLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, lambda_mse: float = 0.2):
        super().__init__()
        self.eps = eps
        self.lambda_mse = lambda_mse

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_2d = pred.permute(0, 2, 1).reshape(-1, pred.shape[1])      # (B*T, C)
        target_2d = target.permute(0, 2, 1).reshape(-1, target.shape[1])  # (B*T, C)

        pred_mean = pred_2d.mean(dim=0, keepdim=True)
        target_mean = target_2d.mean(dim=0, keepdim=True)

        pred_centered = pred_2d - pred_mean
        target_centered = target_2d - target_mean

        numerator = torch.sum(pred_centered * target_centered, dim=0)
        denominator = torch.sqrt(
            torch.sum(pred_centered ** 2, dim=0) *
            torch.sum(target_centered ** 2, dim=0)
        ).clamp(min=self.eps)

        corr = numerator / denominator
        corr_loss = -corr.mean()
        mse_loss = torch.mean((pred - target) ** 2)
        return corr_loss + self.lambda_mse * mse_loss


# =========================
# TCN Model
# =========================

class TemporalConvEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        kernel_size: int,
        out_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()

        if num_layers not in [2, 3, 4]:
            raise ValueError(f"num_layers must be one of [2, 3, 4], got {num_layers}")

        layers: List[nn.Module] = [
            nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
        ]

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
            ])

        layers.append(nn.Conv1d(hidden_dim, out_dim, kernel_size, padding=kernel_size // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        return self.net(x)     # (B, C, T)


class EarlyStopping:
    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - 1e-6:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# =========================
# Seed -> Method Mapping
# =========================

def get_method_config(seed: int) -> Dict:
    if not (0 <= seed < TOTAL_SEEDS):
        raise ValueError(f"seed must be in [0, {TOTAL_SEEDS}), got {seed}")

    if 0 <= seed < 20:
        return {"method": "tcn2_mse", "model_type": "tcn", "num_layers": 2, "loss": "mse"}
    if 20 <= seed < 40:
        return {"method": "tcn3_mse", "model_type": "tcn", "num_layers": 3, "loss": "mse"}
    if 40 <= seed < 60:
        return {"method": "tcn4_mse", "model_type": "tcn", "num_layers": 4, "loss": "mse"}
    if 60 <= seed < 80:
        return {"method": "tcn2_pearson", "model_type": "tcn", "num_layers": 2, "loss": "pearson"}
    if 80 <= seed < 100:
        return {"method": "tcn3_pearson", "model_type": "tcn", "num_layers": 3, "loss": "pearson"}
    if 100 <= seed < 120:
        return {"method": "tcn4_pearson", "model_type": "tcn", "num_layers": 4, "loss": "pearson"}

    return {"method": "bootstrap_ridge", "model_type": "ridge", "num_layers": None, "loss": "ridge"}


def get_seed_list(run_mode: str) -> List[int]:
    if run_mode == "test":
        return TEST_SEEDS.copy()
    if run_mode == "full":
        return list(range(TOTAL_SEEDS))
    raise ValueError(f"Unsupported run_mode for get_seed_list: {run_mode}")


def get_subject_limit(run_mode: str) -> Optional[int]:
    if run_mode in ["test", "ridge_test"]:
        return 2
    return None


def get_ridge_seed_list(run_mode: str) -> List[int]:
    if run_mode == "ridge_test":
        return RIDGE_TEST_SEEDS.copy()
    if run_mode == "ridge_only":
        return RIDGE_FULL_SEEDS.copy()
    raise ValueError(f"Unsupported run_mode for get_ridge_seed_list: {run_mode}")


# =========================
# Atlas and Parcel Labels
# =========================

def decode_label(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def read_schaefer_txt(label_txt_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        label_txt_path,
        sep=r"\s+",
        header=None,
        names=["label", "name", "R", "G", "B", "A"],
    )
    df["name"] = df["name"].apply(decode_label)
    df = df.sort_values("label").reset_index(drop=True)
    df["parcel_index"] = range(len(df))
    return df[["parcel_index", "label", "name", "R", "G", "B", "A"]]


def load_local_schaefer100_atlas() -> Dict:
    if not os.path.exists(ATLAS_NII_PATH):
        raise FileNotFoundError(f"Atlas NIfTI not found: {ATLAS_NII_PATH}")
    if not os.path.exists(ATLAS_TXT_PATH):
        raise FileNotFoundError(f"Atlas TXT not found: {ATLAS_TXT_PATH}")

    label_df = read_schaefer_txt(ATLAS_TXT_PATH)
    if len(label_df) != OUT_CH:
        raise ValueError(f"Expected {OUT_CH} parcels, got {len(label_df)} from atlas txt.")

    return {"maps": ATLAS_NII_PATH, "label_df": label_df}


# =========================
# Feature Loading
# =========================

def extract_feature_array(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().float().numpy()

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, dict):
        if "fusion_embedding" in obj:
            return extract_feature_array(obj["fusion_embedding"])
        raise TypeError(
            f"Loaded feature file is a dict with keys {list(obj.keys())}, "
            "but key 'fusion_embedding' was not found."
        )

    raise TypeError(f"Expected torch.Tensor, np.ndarray, or dict, got {type(obj)}")


def zscore_cols(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def load_feature_matrix(feature_path: str, expected_tr: int) -> np.ndarray:
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    try:
        obj = torch.load(feature_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(feature_path, map_location="cpu")

    feats = extract_feature_array(obj)
    feats = np.asarray(feats, dtype=np.float32)

    if feats.ndim == 1:
        feats = feats[:, None]
    elif feats.ndim > 2:
        feats = feats.reshape(feats.shape[0], -1)

    if feats.shape[0] != expected_tr:
        raise ValueError(
            f"Feature length mismatch for {feature_path}: "
            f"expected {expected_tr}, got {feats.shape[0]}"
        )

    feats = zscore_cols(feats)
    feats = np.nan_to_num(feats).astype(np.float32)

    print(f"[Feature] loaded feature shape = {feats.shape}")
    return feats


def align_feature_matrix_for_ridge(
    feats: np.ndarray,
    raw_tr: int,
    target_tr: int,
    trim_start: int,
    trim_end: int,
) -> np.ndarray:
    """
    Align feature matrix for ridge regression according to fMRI trimming.
    """
    feats = np.asarray(feats, dtype=np.float32)

    if feats.shape[0] == raw_tr:
        end_idx = raw_tr - trim_end if trim_end > 0 else raw_tr
        feats = feats[trim_start:end_idx]
    elif feats.shape[0] == target_tr:
        pass
    elif feats.shape[0] > target_tr:
        warnings.warn(
            f"Feature length {feats.shape[0]} > target TR {target_tr}; truncating to first {target_tr}."
        )
        feats = feats[:target_tr]
    else:
        raise ValueError(
            f"Feature length {feats.shape[0]} is shorter than target TR {target_tr}. "
            "Please check whether the .pt feature file is already TR-aligned."
        )

    feats = zscore_cols(feats)
    feats = np.nan_to_num(feats).astype(np.float32)
    print(f"[Feature] ridge aligned feature shape = {feats.shape}")
    return feats


def make_delayed(stim: np.ndarray, delays, circpad: bool = False) -> np.ndarray:
    nt, ndim = stim.shape
    dstims = []

    for d in delays:
        dstim = np.zeros((nt, ndim), dtype=stim.dtype)
        if d < 0:
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:
            dstim = stim.copy()
        dstims.append(dstim)

    return np.hstack(dstims).astype(np.float32)


# =========================
# fMRI Loading and Trimming
# =========================

def extract_fmri_parcels(
    fmri_path: str,
    atlas_maps: str,
    trim_start: int,
    trim_end: int,
    expected_raw_tr: Optional[int] = None,
    expected_trimmed_tr: Optional[int] = None,
) -> np.ndarray:
    masker = NiftiLabelsMasker(
        labels_img=atlas_maps,
        standardize="zscore_sample",
        strategy="mean",
    )
    ts = masker.fit_transform(fmri_path).astype(np.float32)

    if ts.ndim != 2:
        raise ValueError(f"{os.path.basename(fmri_path)} extracted non-2D array: {ts.shape}")
    if ts.shape[1] != OUT_CH:
        raise ValueError(
            f"{os.path.basename(fmri_path)} extracted {ts.shape[1]} parcels, expected {OUT_CH}"
        )
    if expected_raw_tr is not None and ts.shape[0] != expected_raw_tr:
        raise ValueError(
            f"{os.path.basename(fmri_path)} has {ts.shape[0]} TRs after parcellation, "
            f"expected {expected_raw_tr}"
        )

    end_idx = ts.shape[0] - trim_end if trim_end > 0 else ts.shape[0]
    ts = ts[trim_start:end_idx]

    if expected_trimmed_tr is not None and ts.shape[0] != expected_trimmed_tr:
        raise ValueError(
            f"{os.path.basename(fmri_path)} trimmed TR={ts.shape[0]}, "
            f"expected {expected_trimmed_tr}"
        )

    return ts.astype(np.float32)


def load_all_subject_fmri_fnl(
    fmri_dir: str,
    atlas_maps: str,
    trim_start: int,
    trim_end: int,
    expected_target_tr: int,
    subject_limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[np.ndarray], int, int]:
    fmri_files = sorted(glob.glob(os.path.join(fmri_dir, "*.nii.gz")))
    if len(fmri_files) == 0:
        raise ValueError(f"No .nii.gz files found in {fmri_dir}")

    if subject_limit is not None:
        fmri_files = fmri_files[:subject_limit]

    subject_names: List[str] = []
    all_targets: List[np.ndarray] = []
    raw_tr: Optional[int] = None

    for idx, path in enumerate(fmri_files):
        filename = os.path.basename(path)
        subject_name = filename.split("_", 1)[0]

        if idx == 0:
            masker = NiftiLabelsMasker(
                labels_img=atlas_maps,
                standardize="zscore_sample",
                strategy="mean",
            )
            tmp = masker.fit_transform(path).astype(np.float32)
            raw_tr = tmp.shape[0]

        ts = extract_fmri_parcels(
            fmri_path=path,
            atlas_maps=atlas_maps,
            trim_start=trim_start,
            trim_end=trim_end,
            expected_raw_tr=raw_tr,
            expected_trimmed_tr=expected_target_tr,
        )

        subject_names.append(subject_name)
        all_targets.append(ts)
        print(f"[fMRI] {subject_name} -> raw_TR={raw_tr}, trimmed={ts.shape}")

    assert raw_tr is not None
    return subject_names, fmri_files, all_targets, raw_tr, expected_target_tr


def find_sherlock_fmri_files(fmri_root: str) -> List[str]:
    pattern = os.path.join(
        fmri_root,
        "sub-*",
        "func",
        "*_denoise_crop_smooth6mm_task-sherlockPart1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    fmri_files = sorted(glob.glob(pattern))
    if len(fmri_files) == 0:
        raise ValueError(f"No matching Sherlock Part1 .nii.gz files found under {fmri_root}")
    return fmri_files


def load_all_subject_fmri_sherlock(
    fmri_root: str,
    atlas_maps: str,
    trim_start: int,
    trim_end: int,
    expected_target_tr: int,
    subject_limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[np.ndarray], int, int]:
    fmri_files = find_sherlock_fmri_files(fmri_root)
    if subject_limit is not None:
        fmri_files = fmri_files[:subject_limit]

    subject_names: List[str] = []
    all_targets: List[np.ndarray] = []
    raw_tr: Optional[int] = None

    for idx, path in enumerate(fmri_files):
        subject_name = os.path.basename(os.path.dirname(os.path.dirname(path)))

        if idx == 0:
            masker = NiftiLabelsMasker(
                labels_img=atlas_maps,
                standardize="zscore_sample",
                strategy="mean",
            )
            tmp = masker.fit_transform(path).astype(np.float32)
            raw_tr = tmp.shape[0]

        ts = extract_fmri_parcels(
            fmri_path=path,
            atlas_maps=atlas_maps,
            trim_start=trim_start,
            trim_end=trim_end,
            expected_raw_tr=raw_tr,
            expected_trimmed_tr=expected_target_tr,
        )

        subject_names.append(subject_name)
        all_targets.append(ts)
        print(f"[fMRI] {subject_name} -> raw_TR={raw_tr}, trimmed={ts.shape}")

    assert raw_tr is not None
    return subject_names, fmri_files, all_targets, raw_tr, expected_target_tr


def load_all_subject_fmri(
    config: Dict,
    atlas_maps: str,
    subject_limit: Optional[int] = None,
) -> Tuple[List[str], List[str], List[np.ndarray], int, int]:
    if config["fmri_mode"] == "fnl":
        return load_all_subject_fmri_fnl(
            fmri_dir=config["fmri_dir"],
            atlas_maps=atlas_maps,
            trim_start=config["trim_start"],
            trim_end=config["trim_end"],
            expected_target_tr=config["expected_target_tr"],
            subject_limit=subject_limit,
        )

    if config["fmri_mode"] == "sherlock":
        return load_all_subject_fmri_sherlock(
            fmri_root=config["fmri_root"],
            atlas_maps=atlas_maps,
            trim_start=config["trim_start"],
            trim_end=config["trim_end"],
            expected_target_tr=config["expected_target_tr"],
            subject_limit=subject_limit,
        )

    raise ValueError(f"Unknown fmri_mode: {config['fmri_mode']}")


# =========================
# Data Preparation
# =========================

def prepare_subject_tensors(
    base_feats: np.ndarray,
    all_targets: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for target in all_targets:
        if target.shape[0] != base_feats.shape[0]:
            raise ValueError(
                f"Feature length {base_feats.shape[0]} != target length {target.shape[0]}"
            )
        x_list.append(base_feats.astype(np.float32))
        y_list.append(target.astype(np.float32))

    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)


def split_train_val_indices(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_samples <= 1:
        idx = np.arange(n_samples)
        return idx, idx

    n_train = max(1, int(math.floor(n_samples * 0.8)))
    n_train = min(n_train, n_samples - 1)

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_samples)
    return train_idx, val_idx


# =========================
# Training and Prediction
# =========================

def build_criterion(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "pearson":
        return PearsonCorrLoss()
    raise ValueError(f"Unknown loss_name: {loss_name}")


def train_temporal_model(
    x_train_subjects: np.ndarray,
    y_train_subjects: np.ndarray,
    num_layers: int,
    loss_name: str,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_random_seed(seed)

    train_idx, val_idx = split_train_val_indices(x_train_subjects.shape[0])

    x_train = torch.from_numpy(x_train_subjects[train_idx]).float().to(device)
    y_train = torch.from_numpy(y_train_subjects[train_idx]).float().permute(0, 2, 1).to(device)

    x_val = torch.from_numpy(x_train_subjects[val_idx]).float().to(device)
    y_val = torch.from_numpy(y_train_subjects[val_idx]).float().permute(0, 2, 1).to(device)

    model = TemporalConvEncoder(
        in_dim=x_train.shape[-1],
        hidden_dim=TRAIN_CONFIG["hidden_dim"],
        kernel_size=TRAIN_CONFIG["kernel_size"],
        out_dim=OUT_CH,
        num_layers=num_layers,
    ).to(device)

    criterion = build_criterion(loss_name)
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    stopper = EarlyStopping(patience=TRAIN_CONFIG["patience"])

    for _ in range(TRAIN_CONFIG["max_epochs"]):
        model.train()
        optimizer.zero_grad()

        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val).item()

        if stopper.step(val_loss, model):
            break

    stopper.restore(model)
    model.eval()
    return model


def predict_temporal_model(
    model: nn.Module,
    x_test: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(x_test[None, ...]).float().to(device)
        pred = model(x).squeeze(0).transpose(0, 1).detach().cpu().numpy().astype(np.float32)
    return pred


def normalize_alpha_output(alpha_out) -> float:
    arr = np.asarray(alpha_out).squeeze()
    if arr.ndim == 0:
        return float(arr)
    uniq = np.unique(arr)
    if len(uniq) == 1:
        return float(uniq[0])
    return np.nan


def predict_bootstrap_ridge(
    delayed_feats: np.ndarray,
    train_targets: List[np.ndarray],
    y_test: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Input delayed_feats is already delayed feature matrix.
    """
    X_train = np.vstack([delayed_feats for _ in range(len(train_targets))]).astype(np.float32)
    y_train = np.vstack([np.nan_to_num(y).astype(np.float32) for y in train_targets])

    X_test = delayed_feats.astype(np.float32)
    y_test = np.nan_to_num(y_test).astype(np.float32)

    wt, corr, best_alphas, bscorrs, valinds = bootstrap_ridge(
        np.nan_to_num(X_train),
        np.nan_to_num(y_train),
        np.nan_to_num(X_test),
        np.nan_to_num(y_test),
        ALPHAS,
        NBOOTS,
        CHUNKLEN,
        NCHUNKS,
        singcutoff=1e-10,
        single_alpha=True,
    )

    y_pred = np.dot(X_test, wt).astype(np.float32)
    best_alpha_scalar = normalize_alpha_output(best_alphas)
    return y_pred, best_alpha_scalar


def run_single_seed(
    dataset_name: str,
    seed: int,
    base_feats: np.ndarray,
    subject_names: List[str],
    all_targets: List[np.ndarray],
    device: torch.device,
    ridge_feats: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    cfg = get_method_config(seed)
    seed_predictions: Dict[str, np.ndarray] = {}
    ridge_alpha_map: Dict[str, float] = {}

    X_subjects, Y_subjects = prepare_subject_tensors(base_feats, all_targets)

    loso_bar = tqdm(
        range(len(subject_names)),
        total=len(subject_names),
        desc=f"{dataset_name} seed={seed:03d} {cfg['method']}",
        leave=False,
        dynamic_ncols=True,
    )

    for test_idx in loso_bar:
        test_subject = subject_names[test_idx]
        y_test = np.nan_to_num(all_targets[test_idx]).astype(np.float32)

        train_indices = [i for i in range(len(subject_names)) if i != test_idx]
        train_targets = [all_targets[i] for i in train_indices]

        if cfg["model_type"] == "ridge":
            if ridge_feats is None:
                raise ValueError("ridge_feats must be provided when model_type is ridge.")
            pred, best_alpha = predict_bootstrap_ridge(ridge_feats, train_targets, y_test)
            ridge_alpha_map[test_subject] = best_alpha
        else:
            x_train_subjects = X_subjects[train_indices]
            y_train_subjects = Y_subjects[train_indices]

            model = train_temporal_model(
                x_train_subjects=x_train_subjects,
                y_train_subjects=y_train_subjects,
                num_layers=cfg["num_layers"],
                loss_name=cfg["loss"],
                seed=seed,
                device=device,
            )
            pred = predict_temporal_model(model, base_feats, device)

        seed_predictions[test_subject] = np.nan_to_num(pred).astype(np.float32)

    return seed_predictions, ridge_alpha_map


# =========================
# Ensemble Aggregation
# =========================

def init_sum_store(
    subject_names: List[str],
    template_targets: List[np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    store: Dict[str, Dict[str, np.ndarray]] = {}
    for method in BASE_METHOD_ORDER:
        store[method] = {}
        for subject, target in zip(subject_names, template_targets):
            store[method][subject] = np.zeros_like(target, dtype=np.float64)
    return store


def init_count_store() -> Dict[str, int]:
    return {method: 0 for method in BASE_METHOD_ORDER}


def update_ensemble_store(
    sum_store: Dict[str, Dict[str, np.ndarray]],
    count_store: Dict[str, int],
    method_name: str,
    seed_predictions: Dict[str, np.ndarray],
) -> None:
    for subject, pred in seed_predictions.items():
        sum_store[method_name][subject] += pred
    count_store[method_name] += 1


def finalize_mean_predictions(
    sum_store: Dict[str, Dict[str, np.ndarray]],
    count_store: Dict[str, int],
) -> Dict[str, Dict[str, np.ndarray]]:
    mean_store: Dict[str, Dict[str, np.ndarray]] = {}

    for method, subject_map in sum_store.items():
        if count_store[method] == 0:
            raise ValueError(f"Count for method {method} is zero.")

        mean_store[method] = {}
        for subject, pred_sum in subject_map.items():
            mean_store[method][subject] = (pred_sum / count_store[method]).astype(np.float32)

    return mean_store


# =========================
# Save / Load Predictions
# =========================

def build_prediction_df(pred: np.ndarray, parcel_names: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(pred, columns=parcel_names)
    df.insert(0, "tr", np.arange(pred.shape[0], dtype=int))
    return df


def save_prediction_csv(pred: np.ndarray, parcel_names: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df = build_prediction_df(pred, parcel_names)
    pred_df.to_csv(out_path, index=False, encoding="utf-8")


def load_prediction_csv(pred_csv: Path) -> np.ndarray:
    if not pred_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")
    df = pd.read_csv(pred_csv)
    if "tr" not in df.columns:
        raise ValueError(f"'tr' column not found in {pred_csv}")
    return df.drop(columns=["tr"]).to_numpy(dtype=np.float32)


def save_subject_predictions_base(
    dataset_name: str,
    subject_names: List[str],
    mean_pred_store: Dict[str, Dict[str, np.ndarray]],
    label_df: pd.DataFrame,
) -> None:
    dataset_root = RESULT_ROOT / dataset_name
    subjects_root = dataset_root / "subjects"
    subjects_root.mkdir(parents=True, exist_ok=True)

    parcel_names = label_df["name"].astype(str).tolist()

    for subject in subject_names:
        subject_dir = subjects_root / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        for method in BASE_METHOD_ORDER:
            pred = mean_pred_store[method][subject]
            out_path = subject_dir / METHOD_TO_FILENAME[method]
            save_prediction_csv(pred, parcel_names, out_path)


def refresh_ensembles_from_existing_files(
    dataset_name: str,
    subject_names: List[str],
    label_df: pd.DataFrame,
) -> None:
    """
    Recompute 08_all_model_mean.csv and 09_all_tcn_mean.csv from existing 01~07 files.
    """
    dataset_root = RESULT_ROOT / dataset_name
    subjects_root = dataset_root / "subjects"
    parcel_names = label_df["name"].astype(str).tolist()

    tcn_methods = [
        "tcn2_mse",
        "tcn3_mse",
        "tcn4_mse",
        "tcn2_pearson",
        "tcn3_pearson",
        "tcn4_pearson",
    ]

    all_model_methods = tcn_methods + ["bootstrap_ridge"]

    for subject in subject_names:
        subject_dir = subjects_root / subject

        tcn_preds = []
        for method in tcn_methods:
            pred = load_prediction_csv(subject_dir / METHOD_TO_FILENAME[method])
            tcn_preds.append(pred)

        ridge_pred = load_prediction_csv(subject_dir / METHOD_TO_FILENAME["bootstrap_ridge"])
        all_model_preds = tcn_preds + [ridge_pred]

        all_tcn_mean = np.mean(np.stack(tcn_preds, axis=0), axis=0).astype(np.float32)
        all_model_mean = np.mean(np.stack(all_model_preds, axis=0), axis=0).astype(np.float32)

        save_prediction_csv(
            all_model_mean,
            parcel_names,
            subject_dir / METHOD_TO_FILENAME["all_model_mean"],
        )
        save_prediction_csv(
            all_tcn_mean,
            parcel_names,
            subject_dir / METHOD_TO_FILENAME["all_tcn_mean"],
        )


# =========================
# Evaluation
# =========================

def compute_parcelwise_corrs(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    corrs: List[float] = []
    for p in range(y_true.shape[1]):
        yt = np.asarray(y_true[:, p], dtype=np.float64)
        yp = np.asarray(y_pred[:, p], dtype=np.float64)

        if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
            corrs.append(np.nan)
            continue

        try:
            r = pearsonr(yt, yp)[0]
        except Exception:
            r = np.nan

        corrs.append(float(r) if np.isfinite(r) else np.nan)

    return corrs


def save_all_subject_correlations_100parcels(
    dataset_name: str,
    subject_names: List[str],
    all_targets: List[np.ndarray],
) -> None:
    dataset_root = RESULT_ROOT / dataset_name
    subjects_root = dataset_root / "subjects"

    rows: List[Dict] = []

    for subject, y_true in zip(subject_names, all_targets):
        subject_dir = subjects_root / subject

        for method in ALL_METHOD_ORDER:
            pred = load_prediction_csv(subject_dir / METHOD_TO_FILENAME[method])
            parcel_corrs = compute_parcelwise_corrs(y_true, pred)

            row = {
                "dataset": dataset_name,
                "subject": subject,
                "method": method,
            }

            for p, r in enumerate(parcel_corrs):
                row[f"parcel_{p:03d}"] = r

            finite_mask = np.isfinite(parcel_corrs)
            row["n_valid_parcels"] = int(np.sum(finite_mask))
            row["mean_parcel_pearsonr"] = (
                float(np.nanmean(parcel_corrs)) if np.any(finite_mask) else np.nan
            )

            rows.append(row)

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(
        dataset_root / "all_subject_correlations_100parcels.csv",
        index=False,
        encoding="utf-8",
    )

    summary_df = (
        corr_df.groupby("method", as_index=False)["mean_parcel_pearsonr"]
        .mean()
        .rename(columns={"mean_parcel_pearsonr": "dataset_mean_subject_corr"})
    )
    summary_df.to_csv(
        dataset_root / "dataset_method_summary.csv",
        index=False,
        encoding="utf-8",
    )


# =========================
# Metadata
# =========================

def build_method_counts_from_seeds(executed_seeds: List[int]) -> Dict[str, int]:
    counts = {method: 0 for method in ALL_METHOD_ORDER}

    for seed in executed_seeds:
        method = get_method_config(seed)["method"]
        counts[method] += 1

    counts["all_model_mean"] = (
        counts["tcn2_mse"] +
        counts["tcn3_mse"] +
        counts["tcn4_mse"] +
        counts["tcn2_pearson"] +
        counts["tcn3_pearson"] +
        counts["tcn4_pearson"] +
        counts["bootstrap_ridge"]
    )
    counts["all_tcn_mean"] = (
        counts["tcn2_mse"] +
        counts["tcn3_mse"] +
        counts["tcn4_mse"] +
        counts["tcn2_pearson"] +
        counts["tcn3_pearson"] +
        counts["tcn4_pearson"]
    )
    return counts


def load_existing_method_counts(dataset_name: str) -> Dict[str, int]:
    default_counts = {method: 0 for method in ALL_METHOD_ORDER}
    count_csv = RESULT_ROOT / dataset_name / "method_counts.csv"
    if not count_csv.exists():
        return default_counts

    df = pd.read_csv(count_csv)
    for _, row in df.iterrows():
        method = str(row["method"])
        if method in default_counts:
            default_counts[method] = int(row["n_models"])
    return default_counts


def save_metadata(
    dataset_name: str,
    run_mode: str,
    executed_seeds: List[int],
    overwrite_only_ridge: bool = False,
) -> None:
    dataset_root = RESULT_ROOT / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    mapping_rows = []
    for seed in range(TOTAL_SEEDS):
        cfg = get_method_config(seed)
        mapping_rows.append({
            "seed": seed,
            "executed": int(seed in executed_seeds),
            "run_mode": run_mode,
            "method": cfg["method"],
            "model_type": cfg["model_type"],
            "num_layers": cfg["num_layers"],
            "loss": cfg["loss"],
        })

    pd.DataFrame(mapping_rows).to_csv(
        dataset_root / "seed_method_mapping.csv",
        index=False,
        encoding="utf-8",
    )

    if overwrite_only_ridge:
        counts = load_existing_method_counts(dataset_name)
        ridge_count = len(executed_seeds)
        counts["bootstrap_ridge"] = ridge_count
        counts["all_model_mean"] = (
            counts["tcn2_mse"] +
            counts["tcn3_mse"] +
            counts["tcn4_mse"] +
            counts["tcn2_pearson"] +
            counts["tcn3_pearson"] +
            counts["tcn4_pearson"] +
            counts["bootstrap_ridge"]
        )
        counts["all_tcn_mean"] = (
            counts["tcn2_mse"] +
            counts["tcn3_mse"] +
            counts["tcn4_mse"] +
            counts["tcn2_pearson"] +
            counts["tcn3_pearson"] +
            counts["tcn4_pearson"]
        )
    else:
        counts = build_method_counts_from_seeds(executed_seeds)

    count_rows = [{"method": method, "n_models": counts[method]} for method in ALL_METHOD_ORDER]
    pd.DataFrame(count_rows).to_csv(
        dataset_root / "method_counts.csv",
        index=False,
        encoding="utf-8",
    )


# =========================
# Main Dataset Pipeline (full / test)
# =========================

def run_dataset(
    dataset_name: str,
    device: torch.device,
    run_mode: str = "full",
) -> None:
    if dataset_name not in DATASETS:
        raise ValueError(f"dataset_name must be one of {list(DATASETS.keys())}, got {dataset_name}")
    if run_mode not in ["full", "test"]:
        raise ValueError(f"run_dataset only supports full/test, got {run_mode}")

    config = DATASETS[dataset_name]
    atlas_info = load_local_schaefer100_atlas()
    atlas_maps = atlas_info["maps"]
    label_df = atlas_info["label_df"]

    subject_limit = get_subject_limit(run_mode)
    seed_list = get_seed_list(run_mode)

    base_feats = load_feature_matrix(
        feature_path=config["feature_path"],
        expected_tr=config["expected_feature_tr"],
    )

    subject_names, fmri_files, all_targets, raw_tr, trimmed_tr = load_all_subject_fmri(
        config=config,
        atlas_maps=atlas_maps,
        subject_limit=subject_limit,
    )

    if trimmed_tr != base_feats.shape[0]:
        raise ValueError(
            f"Feature TR ({base_feats.shape[0]}) != fMRI TR ({trimmed_tr}) for dataset {dataset_name}"
        )

    ridge_base_feats = align_feature_matrix_for_ridge(
        feats=base_feats,
        raw_tr=raw_tr,
        target_tr=trimmed_tr,
        trim_start=config["trim_start"],
        trim_end=config["trim_end"],
    )
    ridge_feats = make_delayed(ridge_base_feats, DELAYS)
    ridge_feats = np.nan_to_num(ridge_feats).astype(np.float32)

    print(
        f"[Info] dataset={dataset_name}, subjects={len(subject_names)}, "
        f"raw_tr={raw_tr}, trimmed_tr={trimmed_tr}, feature_shape={base_feats.shape}, "
        f"ridge_feature_shape={ridge_feats.shape}"
    )

    sum_store = init_sum_store(subject_names, all_targets)
    count_store = init_count_store()
    ridge_alpha_rows: List[Dict] = []

    for seed in seed_list:
        set_random_seed(seed)
        cfg = get_method_config(seed)

        seed_predictions, ridge_alpha_map = run_single_seed(
            dataset_name=dataset_name,
            seed=seed,
            base_feats=base_feats,
            subject_names=subject_names,
            all_targets=all_targets,
            device=device,
            ridge_feats=ridge_feats,
        )

        update_ensemble_store(
            sum_store=sum_store,
            count_store=count_store,
            method_name=cfg["method"],
            seed_predictions=seed_predictions,
        )

        if cfg["method"] == "bootstrap_ridge":
            for subject, alpha in ridge_alpha_map.items():
                ridge_alpha_rows.append({
                    "dataset": dataset_name,
                    "subject": subject,
                    "seed": seed,
                    "best_alpha": alpha,
                })

    mean_pred_store = finalize_mean_predictions(sum_store, count_store)

    save_subject_predictions_base(
        dataset_name=dataset_name,
        subject_names=subject_names,
        mean_pred_store=mean_pred_store,
        label_df=label_df,
    )

    refresh_ensembles_from_existing_files(
        dataset_name=dataset_name,
        subject_names=subject_names,
        label_df=label_df,
    )

    save_all_subject_correlations_100parcels(
        dataset_name=dataset_name,
        subject_names=subject_names,
        all_targets=all_targets,
    )

    save_metadata(
        dataset_name=dataset_name,
        run_mode=run_mode,
        executed_seeds=seed_list,
        overwrite_only_ridge=False,
    )

    if len(ridge_alpha_rows) > 0:
        pd.DataFrame(ridge_alpha_rows).to_csv(
            RESULT_ROOT / dataset_name / "ridge_best_alphas.csv",
            index=False,
            encoding="utf-8",
        )


# =========================
# Ridge-only Supplement
# =========================

def run_ridge_only_and_overwrite(
    dataset_name: str,
    device: torch.device,
    run_mode: str = "ridge_only",
) -> None:
    """
    Only run ridge seeds (120-139 or 120 for test) and overwrite bootstrap_ridge results.
    TCN files 01-06 are left untouched. Ensembles 08/09 and evaluation are rebuilt.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"dataset_name must be one of {list(DATASETS.keys())}, got {dataset_name}")
    if run_mode not in ["ridge_only", "ridge_test"]:
        raise ValueError(f"run_ridge_only_and_overwrite only supports ridge_only/ridge_test, got {run_mode}")

    config = DATASETS[dataset_name]
    atlas_info = load_local_schaefer100_atlas()
    atlas_maps = atlas_info["maps"]
    label_df = atlas_info["label_df"]
    parcel_names = label_df["name"].astype(str).tolist()

    subject_limit = get_subject_limit(run_mode)
    ridge_seeds = get_ridge_seed_list(run_mode)

    base_feats = load_feature_matrix(
        feature_path=config["feature_path"],
        expected_tr=config["expected_feature_tr"],
    )

    subject_names, fmri_files, all_targets, raw_tr, trimmed_tr = load_all_subject_fmri(
        config=config,
        atlas_maps=atlas_maps,
        subject_limit=subject_limit,
    )

    ridge_base_feats = align_feature_matrix_for_ridge(
        feats=base_feats,
        raw_tr=raw_tr,
        target_tr=trimmed_tr,
        trim_start=config["trim_start"],
        trim_end=config["trim_end"],
    )
    ridge_feats = make_delayed(ridge_base_feats, DELAYS)
    ridge_feats = np.nan_to_num(ridge_feats).astype(np.float32)

    print(
        f"[RidgeOnly] dataset={dataset_name}, subjects={len(subject_names)}, "
        f"raw_tr={raw_tr}, trimmed_tr={trimmed_tr}, ridge_feature_shape={ridge_feats.shape}, "
        f"n_ridge_seeds={len(ridge_seeds)}"
    )

    dataset_root = RESULT_ROOT / dataset_name
    subjects_root = dataset_root / "subjects"
    subjects_root.mkdir(parents=True, exist_ok=True)

    ridge_sum_store = {
        subject: np.zeros_like(target, dtype=np.float64)
        for subject, target in zip(subject_names, all_targets)
    }
    ridge_alpha_rows: List[Dict] = []

    for seed in ridge_seeds:
        seed_predictions, ridge_alpha_map = run_single_seed(
            dataset_name=dataset_name,
            seed=seed,
            base_feats=ridge_base_feats,
            subject_names=subject_names,
            all_targets=all_targets,
            device=device,
            ridge_feats=ridge_feats,
        )

        for subject in subject_names:
            ridge_sum_store[subject] += seed_predictions[subject]

        for subject, alpha in ridge_alpha_map.items():
            ridge_alpha_rows.append({
                "dataset": dataset_name,
                "subject": subject,
                "seed": seed,
                "best_alpha": alpha,
            })

    ridge_mean_store = {
        subject: (ridge_sum_store[subject] / len(ridge_seeds)).astype(np.float32)
        for subject in subject_names
    }

    for subject in subject_names:
        subject_dir = subjects_root / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        out_path = subject_dir / METHOD_TO_FILENAME["bootstrap_ridge"]
        save_prediction_csv(ridge_mean_store[subject], parcel_names, out_path)
        print(f"[Overwrite] {out_path}")

    if len(ridge_alpha_rows) > 0:
        pd.DataFrame(ridge_alpha_rows).to_csv(
            dataset_root / "ridge_best_alphas.csv",
            index=False,
            encoding="utf-8",
        )

    refresh_ensembles_from_existing_files(
        dataset_name=dataset_name,
        subject_names=subject_names,
        label_df=label_df,
    )

    save_all_subject_correlations_100parcels(
        dataset_name=dataset_name,
        subject_names=subject_names,
        all_targets=all_targets,
    )

    save_metadata(
        dataset_name=dataset_name,
        run_mode=run_mode,
        executed_seeds=ridge_seeds,
        overwrite_only_ridge=True,
    )


# =========================
# Command-line Entry Point
# =========================

def main() -> None:
    """
    Usage:
      python ensemble.py full --dataset fnl
      python ensemble.py test --dataset sherlock
      python ensemble.py ridge_only --dataset fnl
      python ensemble.py ridge_test --dataset sherlock
      python ensemble.py ridge_only --dataset all
    """
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble training for FNL and Sherlock")
    parser.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=["full", "test", "ridge_only", "ridge_test"],
        help="Run mode: full / test / ridge_only / ridge_test. Default is full.",
    )
    parser.add_argument(
        "--dataset",
        choices=["fnl", "sherlock", "all"],
        default="all",
        help="Which dataset to run.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Mode] {args.mode}")
    print(f"[Result Root] {RESULT_ROOT}")

    if args.mode in ["full", "test"]:
        if args.dataset == "all":
            run_dataset("fnl", device, run_mode=args.mode)
            run_dataset("sherlock", device, run_mode=args.mode)
        else:
            run_dataset(args.dataset, device, run_mode=args.mode)
        return

    if args.mode in ["ridge_only", "ridge_test"]:
        if args.dataset == "all":
            run_ridge_only_and_overwrite("fnl", device, run_mode=args.mode)
            run_ridge_only_and_overwrite("sherlock", device, run_mode=args.mode)
        else:
            run_ridge_only_and_overwrite(args.dataset, device, run_mode=args.mode)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
