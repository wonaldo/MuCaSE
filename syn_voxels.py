import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import pearsonr
from nilearn.input_data import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018


# =========================
# Config
# =========================
class Config:
    FMRI_ROOT = "/data"
    N_PARCELS = 100
    ISC_THRESHOLD = 0.2


# =========================
# Atlas Loader
# =========================
def load_schaefer_atlas(n_rois):
    atlas = fetch_atlas_schaefer_2018(n_rois=n_rois)
    atlas_path = atlas["maps"]
    roi_names = atlas["labels"][1:]  # remove background
    return atlas_path, roi_names


# =========================
# fMRI Preprocessing
# =========================
def zscore_fmri(img):
    data = img.get_fdata()
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True) + 1e-8
    z_data = (data - mean) / std
    return nib.Nifti1Image(z_data, img.affine, img.header)


def extract_roi_timeseries(fmri_path, masker):
    img = nib.load(fmri_path)
    z_img = zscore_fmri(img)
    ts = masker.fit_transform(z_img)
    return ts[5:-1]  # temporal trimming


# =========================
# Load Dataset
# =========================
def load_all_subjects(fmri_root, masker, n_rois):
    fmri_files = sorted([
        os.path.join(fmri_root, f)
        for f in os.listdir(fmri_root)
        if f.endswith("_bold.nii.gz")
    ])

    n_subjects = len(fmri_files)
    print(f"Found {n_subjects} subjects.")

    # infer time length dynamically
    sample_ts = extract_roi_timeseries(fmri_files[0], masker)
    T = sample_ts.shape[0]

    all_data = np.zeros((n_subjects, T, n_rois))

    for i, f in enumerate(fmri_files):
        all_data[i] = extract_roi_timeseries(f, masker)

    return all_data, fmri_files


# =========================
# ISC Computation
# =========================
def compute_isc(all_data):
    """
    Compute leave-one-subject-out ISC.
    all_data: [subjects, time, rois]
    """
    n_subjects, _, n_rois = all_data.shape
    mean_isc = np.zeros(n_rois)

    for roi_idx in range(n_rois):
        roi_data = all_data[:, :, roi_idx]
        isc_values = []

        for i in range(n_subjects):
            this_sub = roi_data[i]
            others_mean = roi_data[np.arange(n_subjects) != i].mean(axis=0)

            r, _ = pearsonr(this_sub, others_mean)
            isc_values.append(0 if np.isnan(r) else r)

        mean_isc[roi_idx] = np.mean(isc_values)

    return mean_isc


# =========================
# ROI Selection
# =========================
def select_synchronous_rois(mean_isc, roi_names, threshold):
    mask = mean_isc > threshold
    selected_indices = np.where(mask)[0]
    selected_names = np.array(roi_names)[mask]

    print(f"Synchronous ROIs: {len(selected_indices)} / {len(roi_names)}")

    return selected_indices, selected_names


# =========================
# Save Results
# =========================
def save_isc_results(roi_names, isc_values, save_path):
    df = pd.DataFrame({
        "ROI": roi_names,
        "mean_ISC": isc_values
    })
    df.to_csv(save_path, index=False)
    print(f"ISC results saved to: {save_path}")


# =========================
# Main Pipeline
# =========================
def main():
    cfg = Config()

    # Load atlas
    atlas_path, roi_names = load_schaefer_atlas(cfg.N_PARCELS)

    masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        strategy="mean",
        resampling_target="data"
    )

    # Load all subjects
    all_data, _ = load_all_subjects(cfg.FMRI_ROOT, masker, len(roi_names))

    # Compute ISC
    mean_isc = compute_isc(all_data)

    # Select ROIs
    indices, selected_names = select_synchronous_rois(
        mean_isc, roi_names, cfg.ISC_THRESHOLD
    )

    # Save
    save_isc_results(
        selected_names,
        mean_isc[indices],
        "sync_roi_results.csv"
    )


if __name__ == "__main__":
    main()
