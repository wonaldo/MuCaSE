import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, surface, plotting
import matplotlib.pyplot as plt



# Data Processing
visual_csv = "visual_predominant_pearson.csv"
audio_csv  = "audio_predominant_pearson.csv"


df_vis = pd.read_csv(visual_csv, index_col=0)
df_aud = pd.read_csv(audio_csv, index_col=0)

mean_vis = df_vis.mean(axis=0)
mean_aud = df_aud.mean(axis=0)
ratio = mean_vis / (mean_aud + 1e-8)

# Schaefer 100
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100)
atlas_img = nib.load(schaefer['maps'])
labels = [lab.decode() if isinstance(lab, bytes) else lab for lab in schaefer['labels']]

roi_to_idx = {label.strip().replace("#", "").strip(): idx 
              for idx, label in enumerate(labels)}

atlas_data = atlas_img.get_fdata().astype(int)
out_data = np.zeros_like(atlas_data, dtype=np.float32)

for roi_name, r_val in zip(mean_vis.index, ratio.values):
    clean = roi_name.strip().replace("#", "").strip()
    if clean in roi_to_idx:
        out_data[atlas_data == roi_to_idx[clean]] = r_val

ratio_img = nib.Nifti1Image(out_data, atlas_img.affine)

# Surface Projection
fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
texture_left  = surface.vol_to_surf(ratio_img, fsaverage['pial_left'])
texture_right = surface.vol_to_surf(ratio_img, fsaverage['pial_right'])

# Plot
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), 
                         subplot_kw={'projection': '3d'})

views = ['lateral', 'medial']
hemis = ['left', 'right']
view_names = {'lateral': 'Lateral', 'medial': 'Medial'}
hemi_names = {'left': 'Left', 'right': 'Right'}

for i, hemi in enumerate(hemis):
    for j, view in enumerate(views):
        ax = axes[i, j]
        plotting.plot_surf_stat_map(
            surf_mesh=fsaverage[f'infl_{hemi}'],
            stat_map=texture_left if hemi == 'left' else texture_right,
            hemi=hemi,
            view=view,
            engine='matplotlib',
            cmap='RdYlBu_r',
            colorbar=False,
            threshold=0.75,
            vmin=0.90,
            vmax=1.25,
            bg_map=fsaverage[f'sulc_{hemi}'],
            bg_on_data=True,
            darkness=0.5,
            title=None,
            axes=ax
        )
        ax.set_title(f'{hemi_names[hemi]} - {view_names[view]}', 
                     fontsize=12.5, pad=6)

# ====================== Colorbar ======================
cax = fig.add_axes([0.89, 0.33, 0.011, 0.38]) 
sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0.90, 1.25))
cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=10)

plt.subplots_adjust(left=0.01, right=0.87, bottom=0.04, top=0.96, 
                    wspace=-0.05, hspace=0.03)    

plt.show()