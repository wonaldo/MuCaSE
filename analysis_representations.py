import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### ROI_CLUSTERS
lh_groups = {
    "LH_Vis_Semantic": ['7Networks_LH_Vis_4', '7Networks_LH_Vis_5', '7Networks_LH_Vis_6', '7Networks_LH_Vis_7', '7Networks_LH_Vis_8', '7Networks_LH_Vis_9'],
    "LH_Attn_Control": ['7Networks_LH_DorsAttn_Post_1', '7Networks_LH_DorsAttn_Post_3', '7Networks_LH_DorsAttn_Post_5', '7Networks_LH_DorsAttn_Post_6'],
    "LH_Semantic_Hub": ['7Networks_LH_Default_Temp_1', '7Networks_LH_Default_Par_1', '7Networks_LH_Default_Par_2'],
    "LH_Integration": ['7Networks_LH_Default_pCunPCC_1', '7Networks_LH_Default_pCunPCC_2', '7Networks_LH_Cont_Par_1', '7Networks_LH_Cont_pCun_1'],
    "LH_Affective": ['7Networks_LH_Default_PFC_5', '7Networks_LH_Default_PFC_6', '7Networks_LH_Default_Temp_2', '7Networks_LH_SalVentAttn_Med_2'],
    "RH_Vis_Semantic": ['7Networks_RH_Vis_4', '7Networks_RH_Vis_5', '7Networks_RH_Vis_6', '7Networks_RH_Vis_7', '7Networks_RH_Vis_8'],
    "RH_Attn_Salience": ['7Networks_RH_DorsAttn_Post_1', '7Networks_RH_DorsAttn_Post_3', '7Networks_RH_DorsAttn_Post_4', '7Networks_RH_DorsAttn_Post_5', '7Networks_RH_SalVentAttn_Med_1', '7Networks_RH_SalVentAttn_TempOccPar_1'],
    "RH_Social_Semantic": ['7Networks_RH_Default_Temp_2', '7Networks_RH_Default_Temp_3', '7Networks_RH_Default_Par_1'],
    "RH_Integration": ['7Networks_RH_Default_pCunPCC_1', '7Networks_RH_Default_pCunPCC_2', '7Networks_RH_Cont_Par_2', '7Networks_RH_Cont_pCun_1'],
    "RH_Affective": ['7Networks_RH_Default_PFCv_2', '7Networks_RH_SalVentAttn_Med_1', '7Networks_RH_SalVentAttn_TempOccPar_1']
}


### Model Path
model_paths = {
    "TRIBE_Model": "tribe_features.csv",
    "ImageBind": "imagebind_features.csv",
    "TVLT": "tvlt_features.csv",
    "Causal_Encoding (Ours)":"causal_representations.csv"
}

### Group Mean
lh_res_mean = pd.DataFrame()
for model_name, path in model_paths.items():
    df = pd.read_csv(path, index_col=0)
    m_vals = {}
    for g_name, r_list in lh_groups.items():
        valid_rois = [r for r in r_list if r in df.columns]
        m_vals[g_name] = df[valid_rois].mean(axis=1).mean() if valid_rois else 0.0
    lh_res_mean[model_name] = pd.Series(m_vals)


short_labels = {
    "LH_Vis_Semantic": "LH-Vis",
    "LH_Attn_Control": "LH-Attn",
    "LH_Semantic_Hub": "LH-Sem",
    "LH_Integration": "LH-Int",
    "LH_Affective": "LH-Aff",
    "RH_Vis_Semantic": "RH-Vis",
    "RH_Attn_Salience": "RH-Attn",
    "RH_Social_Semantic": "RH-Soc",
    "RH_Integration": "RH-Int",
    "RH_Affective": "RH-Aff"
}

labels = [short_labels[k] for k in lh_groups.keys()]

### PLOT
plt.figure(figsize=(32, 14))

my_colors = ['#eaf3e2', '#7bc6be', '#439cc4', '#0868a6'] 
# my_colors = ['#f4d9bd', '#eaa975', '#d87636', '#ac491a']

n_groups = len(lh_groups)
n_models = len(model_paths)

group_gap = 5.5
bar_width = 1.0

x = np.arange(n_groups) * group_gap


for i, model in enumerate(lh_res_mean.columns):
    pos = x + (i - (n_models - 1) / 2) * bar_width

    bars = plt.bar(pos, lh_res_mean[model],
                   width=bar_width,
                   label=model,
                   color=my_colors[i],
                   alpha=0.9,
                   edgecolor='white',
                   linewidth=1.2)

    # labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2,
                     height + 0.003,
                     f'{height:.3f}',
                     ha='center',
                     va='bottom',
                     fontsize=10,
                     fontweight='bold')

# xticks
plt.xticks(
    x,
    labels,
    rotation=30,
    ha='right',
    fontsize=24,
    fontweight='semibold'
)

# axis
plt.ylabel("Mean Pearson r", fontsize=28, fontweight='semibold')
plt.yticks(fontsize=22)


plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(
    loc='upper left',
    bbox_to_anchor=(0.78, 1.15),
    frameon=True,
    edgecolor='black',
    fancybox=False,
    prop={'size': 22, 'weight': 'bold'}
)
plt.ylim(0, lh_res_mean.max().max() + 0.05)

plt.tight_layout()

plt.show()