"""Figure: posterior convergence of test users on Smartvote 2023.

Writes figures/posterior_trajectory_smartvote_2023.pdf. Left: the test user
whose final posterior mean lies furthest from the origin under tau^2 = 0.25,
with the posterior after 15 answers, the running posterior mean after every
answer, and the 1-sigma ellipse of the final posterior. Right: CDF over all
test users of the first k at which the running mean enters the final 1-sigma
ellipse, for the tightest and the flattest prior. Also prints, per prior and
weight scaling, the mean, median, and 95th percentile of that k and the
normalised area under the CDF. Reads results/smartvote_2023/latent_convergence/
(written by src.scripts.run_latent_convergence) and the tau^2 = 0.25
checkpoint in results/smartvote_2023/iteration_effect/models/.

Usage:
    python -m src.figures.posterior_trajectory
"""

import ixplore.visualization as ixplore_vis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.patches import Ellipse

from src.data import load_dataset
from src.metrics import gaussian_moments
from src.models.ixplore_wrapper import IXPLOREModel
from src.results import experiment_dir
from src.visualization import apply_defaults, ellipse_legend_handler, make_ellipse, panel_labels, save_figure

DATASET = "smartvote_2023"
TRAJ_DIR = experiment_dir(DATASET, "latent_convergence")
MODELS_DIR = experiment_dir(DATASET, "iteration_effect") / "models"

TAU_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0, 1e6]  # prior variances tau^2
TAU_DEFAULT = 0.25
TAU_TIGHT, TAU_FLAT = 0.05, 1e6
K_SNAP = 15  # answers revealed in the posterior snapshot of the left panel


def load_trajectories(scaled: bool):
    suffix = "_scaled" if scaled else ""
    return {s: pd.read_csv(TRAJ_DIR / f"tau_{s}{suffix}.csv", dtype={"user": str})
            for s in TAU_VALUES}


def first_k_table(t):
    """Per user, the first k at which the running mean is inside the final 1-sigma ellipse."""
    Kt = t["k"].max()
    return (t[t["in_1sigma"]].groupby("user")["k"].min()
            .reindex(t["user"].unique()).fillna(Kt).astype(int))


def make_figure(trajs, per_prior_first_k):
    K = trajs[TAU_DEFAULT]["k"].max()

    # Worst user under the default prior = furthest final posterior mean from origin.
    default_traj = trajs[TAU_DEFAULT]
    final_rows = default_traj[default_traj["k"] == K].copy()
    final_rows["dist"] = np.hypot(final_rows["mean_x"], final_rows["mean_y"])
    worst_user = final_rows.loc[final_rows["dist"].idxmax(), "user"]
    worst_dist = float(final_rows["dist"].max())
    worst_k_to_1sigma = int(per_prior_first_k[TAU_DEFAULT].loc[worst_user])
    print(f"furthest test user (tau^2={TAU_DEFAULT}): {worst_user} "
          f"(||mu_final||={worst_dist:.3f}, k-to-1sigma={worst_k_to_1sigma})")

    test = load_dataset(DATASET, test_fraction=0.15).test_reactions
    worst_inner = IXPLOREModel.load(MODELS_DIR / f"tau_{TAU_DEFAULT}_pca_sp_0.0_iter_20")._model
    worst_answers = test.loc[worst_user]
    worst_final_post = worst_inner.compute_posteriors(worst_answers)
    worst_final_mean, worst_final_cov = gaussian_moments(worst_final_post, worst_inner.X)
    worst_path = (default_traj[default_traj["user"] == worst_user]
                  .sort_values("k")[["mean_x", "mean_y"]].to_numpy())

    # Place axes by hand. The left panel uses set_aspect('equal') on square data,
    # so its allocated box must also be square: panel_h_in == left_w_in. Otherwise
    # the left axes shrinks to a square inside a taller box and the two panels end
    # up unequal heights. fig_h is derived from the panel plus margins, so the
    # saved box exactly contains the panels, the bold letters (margin_t), and the
    # x-labels (margin_b).
    fig_w = 6.0
    left_w_in, right_w_in = 1.85, 3.2
    margin_l, margin_b, margin_t = 0.4, 0.35, 0.12
    gap_in = 0.5
    panel_h_in = left_w_in
    fig_h = panel_h_in + margin_b + margin_t

    fig = plt.figure(figsize=(fig_w, fig_h))
    axL = fig.add_axes([margin_l / fig_w, margin_b / fig_h,
                        left_w_in / fig_w, panel_h_in / fig_h])
    axR = fig.add_axes([(margin_l + left_w_in + gap_in) / fig_w, margin_b / fig_h,
                        right_w_in / fig_w, panel_h_in / fig_h])

    # -- Left panel: posterior snapshot, trajectory, final 1-sigma ellipse --
    n_collections_before = len(axL.collections)
    ixplore_vis.plot_posterior(worst_inner, worst_answers.iloc[:K_SNAP], ax=axL,
                               add_bar=False, cmap=ixplore_vis.colormap)
    for child in axL.collections[n_collections_before:]:
        if isinstance(child, PathCollection) and child.get_offsets().shape[0] == 1:
            child.remove()

    # Trajectory: thin connected line + tiny black dots.
    axL.plot(worst_path[:, 0], worst_path[:, 1], color="black", lw=0.5, alpha=0.8, zorder=4)
    axL.scatter(worst_path[:, 0], worst_path[:, 1], color="black", s=2, zorder=5)
    # Highlights: white circle at k=K_SNAP, coloured star (on top) at the final mean.
    axL.scatter(*worst_path[K_SNAP], facecolor="white", edgecolor="black",
                s=18, lw=0.6, zorder=6, label=f"$k={K_SNAP}$")
    axL.scatter(*worst_final_mean, marker="*", facecolor=ixplore_vis.orange_hex,
                edgecolor="black", s=42, lw=0.4, zorder=8, label=f"$k={K}$")
    ellipse_handle = make_ellipse(worst_final_mean, worst_final_cov, fill=False,
                                  edgecolor="black", lw=0.8, ls="--", zorder=7,
                                  label=r"final 1$\sigma$")
    axL.add_patch(ellipse_handle)

    axL.set_xlim(*worst_inner.limits)
    axL.set_ylim(*worst_inner.limits)
    axL.set_aspect("equal", adjustable="box")
    axL.set_xticks([-1, 0, 1])
    axL.set_yticks([-1, 0, 1])
    axL.set_xlabel("Latent dim 1")
    axL.set_ylabel("Latent dim 2")
    axL.grid(False)
    legL = axL.legend(loc="upper left", title=f"Voter {worst_user}",
                      handlelength=0.9, borderpad=0.25,
                      labelspacing=0.2, handletextpad=0.4, framealpha=1.0,
                      facecolor="white",
                      handler_map={Ellipse: ellipse_legend_handler()})
    legL.get_frame().set_edgecolor("black")
    legL.get_frame().set_linewidth(0.6)

    # -- Right panel: overlay tightest- and flattest-prior CDFs --
    def cdf_for(table):
        sk = np.sort(table.values)
        return sk, np.arange(1, len(sk) + 1) / len(sk)

    sk_tight, cdf_tight = cdf_for(per_prior_first_k[TAU_TIGHT])
    sk_flat, cdf_flat = cdf_for(per_prior_first_k[TAU_FLAT])

    axR.step(sk_tight, cdf_tight, color=ixplore_vis.blue_hex, lw=1.4, where="post",
             label=r"$\tau^2=0.05$ (tight)")
    axR.step(sk_flat, cdf_flat, color=ixplore_vis.orange_hex, lw=1.4, where="post",
             label=r"$\tau^2=10^6$ (flat)")
    axR.fill_between(sk_tight, 0, cdf_tight, step="post", alpha=0.12, color=ixplore_vis.blue_hex)

    axR.axhline(0.5, color="gray", ls=":", lw=0.6)
    axR.axhline(0.95, color="gray", ls=":", lw=0.6)

    # Annotate using the tight-prior CDF.
    k50 = int(sk_tight[np.searchsorted(cdf_tight, 0.5)])
    k95 = int(sk_tight[np.searchsorted(cdf_tight, 0.95)])
    axR.annotate(f"50% at $k={k50}$", xy=(k50, 0.5), xytext=(k50 + 4, 0.40),
                 arrowprops=dict(arrowstyle="->", lw=0.5))
    axR.annotate(f"95% at $k={k95}$", xy=(k95, 0.95), xytext=(k95 - 24, 0.83),
                 arrowprops=dict(arrowstyle="->", lw=0.5))

    # Trim x-axis to where the CDFs reach 1 (no whitespace to the right).
    k_max_reach = int(max(sk_tight[-1], sk_flat[-1]))
    axR.set_xlim(0, k_max_reach)
    axR.set_ylim(0, 1.0)
    axR.set_xlabel("Number of answers $k$")
    axR.set_ylabel(r"Fraction of users inside final 1$\sigma$")
    legR = axR.legend(loc="lower right", handlelength=1.4, borderpad=0.3,
                      labelspacing=0.25, framealpha=1.0, facecolor="white")
    legR.get_frame().set_edgecolor("black")
    legR.get_frame().set_linewidth(0.6)
    panel_labels((axL, axR))
    return fig


def convergence_summary(per_prior_first_k, per_prior_first_k_scaled, K):
    """Mean, median, 95th percentile of k-to-1sigma and AUC = 1 - mean(k)/K per prior."""
    def summarise(table, tau, scaled):
        fk = table.values
        return {
            "tau^2": tau,
            "scale_weights": scaled,
            "mean k": fk.mean().round(1),
            "median k": int(np.median(fk)),
            "k95": int(np.quantile(fk, 0.95)),
            "AUC": round(1 - fk.mean() / K, 3),
        }

    rows = []
    for s in TAU_VALUES:
        rows.append(summarise(per_prior_first_k[s], s, False))
        rows.append(summarise(per_prior_first_k_scaled[s], s, True))
    return pd.DataFrame(rows)


def main():
    apply_defaults()
    trajs = load_trajectories(scaled=False)
    trajs_scaled = load_trajectories(scaled=True)
    per_prior_first_k = {s: first_k_table(t) for s, t in trajs.items()}
    per_prior_first_k_scaled = {s: first_k_table(t) for s, t in trajs_scaled.items()}

    save_figure(make_figure(trajs, per_prior_first_k), f"posterior_trajectory_{DATASET}.pdf")

    K = trajs[TAU_DEFAULT]["k"].max()
    print("\nConvergence summary (k-to-1sigma per prior and weight scaling):")
    print(convergence_summary(per_prior_first_k, per_prior_first_k_scaled, K).to_string(index=False))


if __name__ == "__main__":
    main()
