# tests/test_figures_smoke.py
import pandas as pd
import compute_trajectories as C
import data_io, make_figures as MF

def test_figures_render(tmp_path):
    uids = data_io.test_uids()[:6]
    out = C.run(uids=uids, chunk_size=3, out_dir=tmp_path)
    risk = pd.read_parquet(out["risk_parquet"])
    fig_dir = tmp_path / "figures"
    MF.fig_aggregate_risk(risk, fig_dir)
    MF.fig_window_metrics(risk, fig_dir)
    MF.fig_examples(risk, out["shap_dir"], fig_dir, n_per_class=1)
    assert (fig_dir / "aggregate_risk_vs_lead.pdf").exists()
    assert (fig_dir / "discrimination_vs_lead.pdf").exists()
    assert any(fig_dir.glob("example_*.pdf"))
