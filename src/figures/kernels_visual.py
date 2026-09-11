"""Figure: decision boundaries of the three kernels on one Smartvote 2023 item.

Writes figures/kernels_visual_smartvote_2023.pdf. Picks the item whose
accuracy range across the kernels is largest and shows, one panel per
kernel, the predicted-probability surface of that item over the latent
space with the candidates coloured by their actual response. Loads the
kernel checkpoints from results/smartvote_2023/feature_effect/models/.

Usage:
    python -m src.figures.kernels_visual
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ixplore.visualization import colormap as ixplore_cmap
from matplotlib import cm
from matplotlib.colors import Normalize

from src.data import DATASET_DIRS, load_dataset
from src.models.ixplore_wrapper import IXPLOREModel
from src.results import experiment_dir
from src.visualization import apply_defaults, figsize, panel_labels, save_figure

DATASET = "smartvote_2023"
MODELS_DIR = experiment_dir(DATASET, "feature_effect") / "models"
KERNELS = ["linear", "polynomial", "rff"]
KERNEL_LABELS = {"linear": "Linear", "polynomial": "Polynomial", "rff": "RFF"}

# Metric used to pick the most kernel-discriminating item: "accuracy" or "mae".
SELECTION_METRIC = "accuracy"

# Smartvote 5-point agreement scale.
AGREEMENT_LABELS = {
    0.0: "Disagree",
    0.25: "Rather disagree",
    0.5: "Neutral",
    0.75: "Rather agree",
    1.0: "Agree",
}


def select_item(models, reactions):
    """Per-item MAE and accuracy per kernel; return the item with the largest spread."""
    per_item_mae, per_item_acc = {}, {}
    preds_by_kernel, embeddings_by_kernel = {}, {}
    for name, model in models.items():
        emb = model.get_embedding().values
        preds = model.predict(emb)
        actual = reactions.values.astype(float)
        mask = ~np.isnan(actual)
        abs_err = np.where(mask, np.abs(actual - preds), 0.0)
        correct = np.where(mask, (preds >= 0.5) == (actual >= 0.5), 0.0)
        counts = mask.sum(axis=0)
        per_item_mae[name] = np.where(counts > 0, abs_err.sum(axis=0) / np.maximum(counts, 1), np.nan)
        per_item_acc[name] = np.where(counts > 0, correct.sum(axis=0) / np.maximum(counts, 1), np.nan)
        preds_by_kernel[name] = preds
        embeddings_by_kernel[name] = emb

    mae_matrix = pd.DataFrame(per_item_mae, index=reactions.columns)
    acc_matrix = pd.DataFrame(per_item_acc, index=reactions.columns)
    selection_source = {"accuracy": acc_matrix, "mae": mae_matrix}[SELECTION_METRIC]
    score_range = selection_source.max(axis=1) - selection_source.min(axis=1)
    selected_item = score_range.idxmax()

    print(f"Selection metric: {SELECTION_METRIC}")
    print(f"Item with largest spread across kernels: {selected_item}")
    print(f"MAE per kernel:      {dict(mae_matrix.loc[selected_item].round(3))}")
    print(f"Accuracy per kernel: {dict(acc_matrix.loc[selected_item].round(3))}")
    return selected_item, preds_by_kernel, embeddings_by_kernel


def make_figure(models, reactions, selected_item, preds_by_kernel, embeddings_by_kernel):
    selected_idx = list(reactions.columns).index(selected_item)
    actual_item = reactions.iloc[:, selected_idx].values
    mask_item = ~np.isnan(actual_item)
    value_counts = pd.Series(actual_item[mask_item]).value_counts().sort_index()

    resolution = 100
    grid_x = np.linspace(-1, 1, resolution)
    grid_y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # 3 square panels at textwidth: ~2 in / panel, plus padding.
    fig, axes = plt.subplots(1, 3, figsize=figsize(width_frac=1.0, aspect=0.32))

    # Panel order for this figure (reversed relative to KERNELS).
    panel_kernels = list(reversed(KERNELS))

    for ax, name in zip(axes, panel_kernels):
        model = models[name]
        emb = embeddings_by_kernel[name]
        preds_grid = model.predict(grid_points)
        zz = preds_grid[:, selected_idx].reshape(xx.shape)
        # Flip across the y-axis: mirror both the contour grid and the embedding x.
        ax.contourf(-xx, yy, zz, levels=20, cmap=ixplore_cmap, vmin=0, vmax=1)

        candidate_colors = np.array([ixplore_cmap(v) if not np.isnan(v) else (0.6, 0.6, 0.6, 1.0)
                                     for v in actual_item.astype(float)])
        ax.scatter(-emb[:, 0], emb[:, 1], c=candidate_colors, s=10,
                   edgecolors="white", linewidths=0.3, zorder=2)

        item_preds = preds_by_kernel[name][:, selected_idx]
        mae = np.mean(np.abs(actual_item[mask_item] - item_preds[mask_item]))
        acc = np.mean((item_preds[mask_item] >= 0.5) == (actual_item[mask_item] >= 0.5))

        bbox = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                    edgecolor="0.5", linewidth=0.6)
        ax.text(0.04, 0.04,
                f"$\\bf{{{KERNEL_LABELS[name]}}}$\nMAE={mae:.3f}\nACC={acc * 100:.1f}%",
                transform=ax.transAxes, ha="left", va="bottom",
                linespacing=1.4, bbox=bbox)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(False)

    # Shared agreement-value legend at the top of the middle panel.
    agreement_values = sorted(value_counts.index.tolist())
    agreement_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                    markerfacecolor=ixplore_cmap(v), markersize=4,
                                    markeredgecolor="white", markeredgewidth=0.3,
                                    label=AGREEMENT_LABELS.get(round(v, 2), f"{v:.2f}"))
                         for v in agreement_values]
    leg = axes[1].legend(handles=agreement_handles, loc="upper center",
                         ncol=len(agreement_values), fontsize=7,
                         handlelength=0.6, handletextpad=0.1,
                         columnspacing=0.5, labelspacing=0.2,
                         borderpad=0.3, framealpha=1.0, facecolor="white")
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)
    axes[1].set_zorder(axes[0].get_zorder() + 1)
    axes[1].set_zorder(axes[2].get_zorder() + 1)
    axes[1].patch.set_visible(False)
    leg.set_zorder(20)

    panel_labels(axes)
    fig.subplots_adjust(top=0.95, bottom=0.005, left=0.005, right=0.9, wspace=0.05)

    # Thin shared colorbar matched to the vertical extent of panel C.
    panel_bbox = axes[2].get_position()
    cbar_ax = fig.add_axes([0.92, panel_bbox.y0, 0.01, panel_bbox.height])
    cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap=ixplore_cmap), cax=cbar_ax)
    cbar.set_label("Predicted probability", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.outline.set_linewidth(0.4)
    return fig


def main():
    apply_defaults()
    reactions = load_dataset(DATASET).train_reactions
    statements = pd.read_csv(DATASET_DIRS[DATASET] / "statements.csv").set_index("ID_question")
    models = {name: IXPLOREModel.load(MODELS_DIR / f"{name}_sp_0.0") for name in KERNELS}

    selected_item, preds_by_kernel, embeddings_by_kernel = select_item(models, reactions)
    print(f"Question (EN): {statements.loc[int(selected_item), 'question_EN']}")

    fig = make_figure(models, reactions, selected_item, preds_by_kernel, embeddings_by_kernel)
    save_figure(fig, f"kernels_visual_{DATASET}.pdf")


if __name__ == "__main__":
    main()
