# window_metrics.py — discrimination as a function of lead time + headline averages.
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from config import RISK_PARQUET, WINDOW_METRICS


def compute(risk: pd.DataFrame):
    """Return (per_k DataFrame[k,n_patients,n_pos,AUPRC,AUROC], overall dict)."""
    per = []
    for k, g in risk.groupby("k"):
        y, p = g["y_true"].to_numpy(), g["ensemble_risk"].to_numpy()
        npos = int((y == 1).sum())
        auprc = average_precision_score(y, p) if 0 < npos < len(y) else np.nan
        auroc = roc_auc_score(y, p) if 0 < npos < len(y) else np.nan
        per.append({"k": int(k), "n_patients": len(g), "n_pos": npos,
                    "AUPRC": auprc, "AUROC": auroc})
    per_k = pd.DataFrame(per).sort_values("k").reset_index(drop=True)
    overall = {
        "auprc_pooled": float(average_precision_score(risk["y_true"], risk["ensemble_risk"])),
        "auroc_pooled": float(roc_auc_score(risk["y_true"], risk["ensemble_risk"])),
        "auprc_macro": float(per_k["AUPRC"].mean(skipna=True)),
        "auroc_macro": float(per_k["AUROC"].mean(skipna=True)),
        "n_window_rows": int(len(risk)),
    }
    return per_k, overall


def run(risk_parquet=None, out_csv=None):
    risk = pd.read_parquet(RISK_PARQUET if risk_parquet is None else risk_parquet)
    per_k, overall = compute(risk)
    out_csv = WINDOW_METRICS if out_csv is None else out_csv
    per_k.to_csv(out_csv, index=False)
    # persist the headline averages (overall AUPRC over all windows) next to the per-k CSV
    (out_csv.parent / "window_metrics_summary.json").write_text(json.dumps(overall, indent=2))
    print("Headline:", {k: round(v, 4) if isinstance(v, float) else v for k, v in overall.items()})
    print(f"k=0 AUPRC={per_k.loc[per_k.k==0,'AUPRC'].iloc[0]:.4f} "
          f"AUROC={per_k.loc[per_k.k==0,'AUROC'].iloc[0]:.4f} (expect ~0.7913 / 0.9229)")
    return per_k, overall


if __name__ == "__main__":
    run()
