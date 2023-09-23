import os
import pandas as pd
from rhythmic_relationships import DATASETS_DIR, PLOTS_DIRNAME, MODELS_DIR
from rhythmic_relationships.model_utils import load_model
from rhythmic_relationships.models.hits_encdec import TransformerEncoderDecoder

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

sns.set_style("white")
sns.set_context("paper")

# fig size in inches
rcParams['axes.titlesize'] = 16
rcParams["figure.figsize"] = [6, 4]  # fig size in inches
rcParams["figure.dpi"] = 300

model_type = "hits_encdec"

# Melody -> Bass
mtb_model_name = "fragmental_2306210056"
mtb_model_dir = os.path.join(MODELS_DIR, model_type, mtb_model_name)
mtb_model_eval_dir = os.path.join(mtb_model_dir, "eval")
mtb_model_path = os.path.join(mtb_model_dir, "model.pt")

_, mtb_config = load_model(mtb_model_path, TransformerEncoderDecoder)
dataset_plots_dir = os.path.join(
    DATASETS_DIR, mtb_config["data"]["dataset_name"], PLOTS_DIRNAME
)

mtb_paired_df = pd.read_csv(
    os.path.join(mtb_model_eval_dir, f"src_gen_paired_desc_dists_100_nucleus.csv")
)

# Bass -> Melody
btm_model_name = "literation_2307011858"
btm_model_dir = os.path.join(MODELS_DIR, model_type, btm_model_name)
btm_model_eval_dir = os.path.join(btm_model_dir, "eval")
btm_model_path = os.path.join(btm_model_dir, "model.pt")

_, config = load_model(btm_model_path, TransformerEncoderDecoder)

btm_paired_df = pd.read_csv(
    os.path.join(btm_model_eval_dir, f"src_gen_paired_desc_dists_100_nucleus.csv")
)

# Baseline
baseline_dir = os.path.join(DATASETS_DIR, "baseline")
baseline_paired_df = pd.read_csv(
    os.path.join(baseline_dir, f"baseline_mvae_paired_descs.csv")
)


# Reference
ref_paired_df = pd.read_csv(
    os.path.join(dataset_plots_dir, "paired_melody_bass_descriptors.csv")
)

# Plot
for feature in ref_paired_df.columns:
    ref = ref_paired_df[[feature]]
    base = baseline_paired_df[[feature]]
    mtb = mtb_paired_df[[feature]]
    btm = btm_paired_df[[feature]]

    ref.columns = ["Train"]
    base.columns = ["Baseline"]
    mtb.columns = ["Melody to Bass"]
    btm.columns = ["Bass to Melody"]

    compare_df = pd.concat(
        [
            ref,
            base,
            mtb,
            btm,
        ]
    )

    # Asymmetrical violin plot
    ax = sns.violinplot(
        data=pd.melt(
            compare_df,
            value_vars=["Train", "Baseline", "Melody to Bass", "Bass to Melody"],
        ),
        x="variable",
        y="value",
    )
    plt.ylabel("")
    plt.xlabel("")
    plt.yticks([])
    title = "Onset Balance"
    if feature == 'antiphony':
        title = 'Antiphony'
    plt.title(title)
    plt.legend("")
    plt.tight_layout()
    outpath = os.path.join(dataset_plots_dir, f"all_paired_eval_{feature}.png")
    plt.savefig(outpath)
    plt.clf()
    print(f"Saved {outpath}")


# Plotting the distributions of onset_balance and antiphony for the four sources
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plotting onset_balance
sns.kdeplot(ref_paired_df['onset_balance'], label='Training', ax=axes[0], color='b')
sns.kdeplot(baseline_paired_df['onset_balance'], label='Baseline', ax=axes[0], color='r')
sns.kdeplot(mtb_paired_df['onset_balance'], label='Melody to Bass (MTB)', ax=axes[0], color='g')
sns.kdeplot(btm_paired_df['onset_balance'], label='Bass to Melody (BTM)', ax=axes[0], color='y')
axes[0].set_title('Onset Balance', fontsize=16)
axes[0].legend(fontsize=14)
axes[0].set_ylabel('Density', fontsize=16)
axes[0].set_xlabel('')
axes[0].set_yticklabels('')
axes[0].tick_params(axis='both', which='major', labelsize=12)

# Plotting antiphony
sns.kdeplot(ref_paired_df['antiphony'], label='Training', ax=axes[1], color='b')
sns.kdeplot(baseline_paired_df['antiphony'], label='Baseline', ax=axes[1], color='r')
sns.kdeplot(mtb_paired_df['antiphony'], label='Melody to Bass (MTB)', ax=axes[1], color='g')
sns.kdeplot(btm_paired_df['antiphony'], label='Bass to Melody (BTM)', ax=axes[1], color='y')
axes[1].set_title('Antiphony', fontsize=16)
axes[1].legend(fontsize=14)
axes[1].set_xlabel('')
axes[1].set_ylabel('')
axes[1].set_yticklabels('')
axes[1].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
outpath_2 = os.path.join(dataset_plots_dir, f"all_paired_eval_2.png")
plt.savefig(outpath_2)
plt.clf()
print(f"Saved {outpath_2}")
plt.show()
