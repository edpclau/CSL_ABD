# make_figures.py — risk/SHAP trajectory figures from artifacts.
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from window_metrics import compute
from config import FIG_DIR, COMPONENTS


def _ensure(fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def fig_aggregate_risk(risk: pd.DataFrame, fig_dir=FIG_DIR):
    fig_dir = _ensure(fig_dir)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for y, label, color in [(1, "Case (Outcome=1)", "crimson"), (0, "Control", "steelblue")]:
        g = risk[risk["y_true"] == y]
        stat = g.groupby("k")["ensemble_risk"].agg(["mean", "count",
              lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        stat.columns = ["mean", "count", "q25", "q75"]
        stat = stat[stat["count"] >= 10]
        ax.plot(stat.index, stat["mean"], color=color, label=label)
        ax.fill_between(stat.index, stat["q25"], stat["q75"], color=color, alpha=0.15)
    ax.set_xlabel("hours before anchor (k; 0 = tested window)")
    ax.set_ylabel("mean ensemble risk")
    ax.set_title("Risk trajectory vs lead time")
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "aggregate_risk_vs_lead.pdf"); plt.close(fig)


def fig_window_metrics(risk: pd.DataFrame, fig_dir=FIG_DIR):
    fig_dir = _ensure(fig_dir)
    per_k, _ = compute(risk)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(per_k["k"], per_k["AUPRC"], color="darkgreen", label="AUPRC")
    ax1.plot(per_k["k"], per_k["AUROC"], color="darkorange", label="AUROC")
    ax1.set_xlabel("hours before anchor (k)")
    ax1.set_ylabel("discrimination")
    ax2 = ax1.twinx()
    ax2.fill_between(per_k["k"], per_k["n_patients"], color="gray", alpha=0.12)
    ax2.set_ylabel("n patients at offset (length bias)")
    ax1.legend(loc="lower left")
    ax1.set_title("Per-window discrimination vs lead time")
    fig.tight_layout(); fig.savefig(fig_dir / "discrimination_vs_lead.pdf"); plt.close(fig)


def _pick_examples(risk, n_per_class):
    k0 = risk[risk["k"] == 0].copy()
    out = {}
    out["TP"] = k0[(k0.y_true == 1) & (k0.ensemble_risk >= 0.5)].nlargest(n_per_class, "ensemble_risk")
    out["FN"] = k0[(k0.y_true == 1) & (k0.ensemble_risk < 0.5)].nsmallest(n_per_class, "ensemble_risk")
    out["FP"] = k0[(k0.y_true == 0) & (k0.ensemble_risk >= 0.5)].nlargest(n_per_class, "ensemble_risk")
    out["TN"] = k0[(k0.y_true == 0) & (k0.ensemble_risk < 0.5)].nsmallest(n_per_class, "ensemble_risk")
    return out


def fig_examples(risk, shap_dir, fig_dir=FIG_DIR, n_per_class=1, top_k=6):
    fig_dir = _ensure(fig_dir)
    names = json.loads((shap_dir / "feature_names.json").read_text())
    head_key = {"EEG": "eeg", "CT": "ct", "MRI": "mri"}
    for cls, sel in _pick_examples(risk, n_per_class).items():
        for uid in sel["uid"]:
            rg = risk[risk["uid"] == uid].sort_values("k")
            z = np.load(shap_dir / f"{uid}.npz")
            order = np.argsort(z["k"])
            ks = z["k"][order]
            # choose the head whose component prob is largest at k=0 for this patient
            comp_at0 = rg[rg["k"] == 0].iloc[0][["EEG_p", "CT_p", "MRI_p"]].to_numpy()
            head = COMPONENTS[int(np.argmax(comp_at0))]
            S = z[head_key[head]][order][:, :811]                 # drop bias col
            top = np.argsort(np.abs(S).max(axis=0))[::-1][:top_k]

            fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1.4]})
            a1.plot(rg["k"], rg["ensemble_risk"], color="black", lw=2)
            a1.axhline(0.5, color="gray", ls="--", lw=0.8)
            a1.set_ylabel("ensemble risk"); a1.set_title(f"{cls}  uid={uid}  (SHAP head: {head})")
            for j in top:
                a2.plot(ks, S[:, j], label=names[j])
            a2.axhline(0, color="gray", lw=0.6)
            a2.set_xlabel("hours before anchor (k; 0 = tested window)")
            a2.set_ylabel(f"{head} head SHAP (logit)")
            a2.legend(fontsize=7, ncol=2)
            a1.invert_xaxis()                                     # admission -> anchor left-to-right
            fig.tight_layout(); fig.savefig(fig_dir / f"example_{cls}_{uid}.pdf"); plt.close(fig)
