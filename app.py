from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DEFAULT_DASHBOARD_MODEL, OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR, RUNS_DIR, TELEMETRY_DIR, VERIFY_DIR


st.set_page_config(
    page_title="F1 Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataframe(path: Path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Manrope:wght@400;600;700&display=swap');
        :root {
            --bg-1: #030712;
            --bg-2: #07101f;
            --panel: rgba(9, 14, 29, 0.72);
            --panel-border: rgba(0, 255, 234, 0.25);
            --text-main: #f7fbff;
            --text-soft: #93b3c8;
            --accent-a: #00f5d4;
            --accent-b: #7b2ff7;
            --accent-c: #ff4ecd;
            --accent-d: #2cf6ff;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 10% 20%, rgba(0, 245, 212, 0.16), transparent 30%),
                radial-gradient(circle at 90% 10%, rgba(123, 47, 247, 0.18), transparent 25%),
                radial-gradient(circle at 80% 90%, rgba(255, 78, 205, 0.16), transparent 20%),
                linear-gradient(120deg, var(--bg-1), var(--bg-2));
            color: var(--text-main);
            font-family: 'Manrope', sans-serif;
        }
        [data-testid="stSidebar"] {
            background: rgba(3, 9, 20, 0.92);
            border-right: 1px solid rgba(0,255,234,0.08);
        }
        h1, h2, h3, .neon-title {
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 0.04em;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 1.4rem 1.6rem 1.6rem 1.6rem;
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(0,245,212,0.10), rgba(123,47,247,0.12)),
                rgba(8, 14, 30, 0.78);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 0 0 1px rgba(0,245,212,0.08), 0 25px 80px rgba(0,0,0,0.45);
            backdrop-filter: blur(16px);
            animation: pulseGlow 7s ease-in-out infinite alternate;
        }
        .hero-shell::before {
            content: "";
            position: absolute;
            inset: -25%;
            background: conic-gradient(from 120deg, transparent, rgba(0,245,212,0.15), transparent, rgba(255,78,205,0.15), transparent);
            animation: spinSlow 18s linear infinite;
        }
        .hero-inner { position: relative; z-index: 2; }
        .hero-kicker {
            text-transform: uppercase;
            font-size: 0.8rem;
            color: var(--accent-a);
            letter-spacing: 0.22em;
            margin-bottom: 0.6rem;
        }
        .hero-title {
            font-size: 2.5rem;
            line-height: 1;
            margin: 0 0 0.7rem 0;
            color: #fcffff;
            text-shadow: 0 0 18px rgba(44,246,255,0.45);
        }
        .hero-copy {
            color: var(--text-soft);
            max-width: 820px;
            font-size: 1rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-top: 1.2rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 32px rgba(0,0,0,0.22);
            transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
        }
        .metric-card:hover {
            transform: translateY(-4px) scale(1.01);
            border-color: rgba(0,245,212,0.28);
            box-shadow: 0 0 0 1px rgba(0,245,212,0.10), 0 20px 50px rgba(0,0,0,0.35);
        }
        .metric-label {
            color: var(--text-soft);
            font-size: 0.86rem;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 800;
            color: #fff;
        }
        .section-card {
            background: var(--panel);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            padding: 1rem 1rem 0.4rem 1rem;
            box-shadow: 0 18px 48px rgba(0,0,0,0.28);
            backdrop-filter: blur(14px);
            margin-bottom: 1rem;
        }
        .micro-note {
            color: var(--text-soft);
            font-size: 0.88rem;
        }
        @keyframes pulseGlow {
            from { box-shadow: 0 0 0 1px rgba(0,245,212,0.08), 0 25px 80px rgba(0,0,0,0.45); }
            to { box-shadow: 0 0 0 1px rgba(123,47,247,0.14), 0 25px 100px rgba(0,0,0,0.52); }
        }
        @keyframes spinSlow {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 0.8rem;
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metrics_report: dict, predictions_df: pd.DataFrame, race_df: pd.DataFrame, selected_race: str, selected_model: str):
    best_model = min(metrics_report, key=lambda name: metrics_report[name]["mae"]) if metrics_report else "n/a"
    pod_prob = race_df["podium_probability"].max() if "podium_probability" in race_df else 0
    model_count = len(metrics_report)
    predicted_winner = "n/a"
    if not race_df.empty and f"pred_{selected_model}" in race_df.columns:
        predicted_winner = race_df.nsmallest(1, f"pred_{selected_model}")["driver"].iloc[0]

    st.markdown(
        f"""
        <div class="hero-shell">
          <div class="hero-inner">
            <div class="hero-kicker">Race Intelligence Command Center</div>
            <div class="hero-title">F1 Race Intelligence</div>
            <div class="hero-copy">
              Interactive race view for <strong>{selected_race}</strong> using the <strong>{selected_model}</strong> predictor.
              The cards below summarize what the current model thinks will happen in this selected race.
            </div>
            <div class="metric-grid">
              <div class="metric-card"><div class="metric-label">Selected Model</div><div class="metric-value">{selected_model.upper()}</div></div>
              <div class="metric-card"><div class="metric-label">Selected Race</div><div class="metric-value">{selected_race}</div></div>
              <div class="metric-card"><div class="metric-label">Predicted Winner</div><div class="metric-value">{predicted_winner}</div></div>
              <div class="metric-card"><div class="metric-label">Peak Podium Probability</div><div class="metric-value">{pod_prob:.0%}</div></div>
              <div class="metric-card"><div class="metric-label">Best Overall Model</div><div class="metric-value">{best_model.upper()}</div></div>
              <div class="metric-card"><div class="metric-label">Tracked Models</div><div class="metric-value">{model_count}</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_band(run_manifest: dict, run_summary: dict, leakage_audit: dict, verify_report: dict):
    started = run_manifest.get("started_at", "n/a")
    finished = run_manifest.get("finished_at", "running")
    runtime = run_manifest.get("runtime_seconds", "n/a")
    leakage = leakage_audit.get("status", "unknown")
    verify_required = verify_report.get("required_modules", {})
    missing_required = sum(1 for value in verify_required.values() if value != "ok")

    st.markdown(
        f"""
        <div class="section-card">
          <div class="hero-inner">
            <div class="hero-kicker">Run Health</div>
            <div class="metric-grid">
              <div class="metric-card"><div class="metric-label">Started</div><div class="metric-value">{started}</div></div>
              <div class="metric-card"><div class="metric-label">Finished</div><div class="metric-value">{finished}</div></div>
              <div class="metric-card"><div class="metric-label">Runtime (s)</div><div class="metric-value">{runtime}</div></div>
              <div class="metric-card"><div class="metric-label">Leakage Audit</div><div class="metric-value">{str(leakage).upper()}</div></div>
              <div class="metric-card"><div class="metric-label">Verify Setup Issues</div><div class="metric-value">{missing_required}</div></div>
              <div class="metric-card"><div class="metric-label">Ablation Rows</div><div class="metric-value">{run_summary.get("ablation_rows", 0)}</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_orientation_panel():
    st.markdown(
        """
        <div class="section-card">
          <div class="hero-inner">
            <div class="hero-kicker">What Am I Looking At?</div>
            <div class="hero-copy">
              This dashboard is a race-by-race ML analysis console. The left controls choose one Formula 1 race and one prediction model.
              The tabs then update to show how that model ranked drivers, what probabilities it assigned, and which supporting charts explain the result.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_workflow():
    st.markdown(
        """
        <div class="section-card">
          <div class="hero-inner">
            <div class="hero-kicker">How Prediction Works</div>
            <div class="hero-copy">
              The predictor learns from historical race weekends, engineers features such as grid position, qualifying gap, recent form,
              track fit, strategy context, and DNF risk, then compares multiple models. The selected model produces finishing-order scores,
              while separate probability heads estimate podium, points, and DNF likelihood. The explainer layer then turns those numbers
              into race visuals, feature summaries, and telemetry-backed support plots.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selected_race_summary(race_df: pd.DataFrame, selected_race: str, selected_model: str):
    if race_df.empty or f"pred_{selected_model}" not in race_df.columns:
        return

    predicted_winner_row = race_df.nsmallest(1, f"pred_{selected_model}").iloc[0]
    top3_df = race_df.nsmallest(3, f"pred_{selected_model}")
    podium_names = ", ".join(top3_df["driver"].tolist())
    highest_dnf_row = race_df.nlargest(1, "dnf_probability").iloc[0]

    summary = (
        f"For {selected_race}, the {selected_model.upper()} model predicts {predicted_winner_row['driver']} as the most likely winner. "
        f"Its predicted podium is {podium_names}. "
        f"{predicted_winner_row['driver']} carries a podium probability of {predicted_winner_row['podium_probability']:.1%}, "
        f"while {highest_dnf_row['driver']} has the highest DNF risk at {highest_dnf_row['dnf_probability']:.1%}. "
        f"Use Race Pulse for the full driver ordering and Explanations for feature-level reasoning."
    )

    st.markdown(
        f"""
        <div class="section-card">
          <div class="hero-inner">
            <div class="hero-kicker">Selected Race Summary</div>
            <div class="hero-copy">{summary}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_latest_run_manifest():
    manifests = sorted(RUNS_DIR.glob("run_*/manifest.json"))
    if not manifests:
        return {}
    return load_json(manifests[-1], {})


def build_ablation_chart(ablation_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ablation_df["stage"],
        y=ablation_df["mae"],
        mode="lines+markers",
        name="MAE",
        line=dict(color="#00f5d4", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=ablation_df["stage"],
        y=ablation_df["spearman_r"],
        mode="lines+markers",
        name="Spearman",
        line=dict(color="#ff4ecd", width=3),
        yaxis="y2",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        title="Ablation study progress",
        yaxis=dict(title="MAE"),
        yaxis2=dict(title="Spearman", overlaying="y", side="right"),
    )
    return fig


def build_model_bar(metrics_report: dict):
    df = pd.DataFrame(metrics_report).T.reset_index().rename(columns={"index": "model"})
    fig = go.Figure()
    fig.add_trace(go.Bar(name="MAE", x=df["model"], y=df["mae"], marker_color="#00f5d4"))
    fig.add_trace(go.Bar(name="RMSE", x=df["model"], y=df["rmse"], marker_color="#ff4ecd"))
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        title="Model comparison",
    )
    return fig


def build_probability_chart(race_df: pd.DataFrame):
    top = race_df.sort_values("podium_probability", ascending=False).head(10)
    fig = px.bar(
        top,
        x="podium_probability",
        y="driver",
        orientation="h",
        color="podium_probability",
        color_continuous_scale=["#09121f", "#00f5d4", "#ff4ecd"],
        template="plotly_dark",
        title="Podium pressure map",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def build_finish_order_chart(race_df: pd.DataFrame, selected_model: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=race_df["driver"],
        y=race_df["actual_finish"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#00f5d4", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=race_df["driver"],
        y=race_df[f"pred_{selected_model}"],
        mode="lines+markers",
        name=f"Predicted ({selected_model})",
        line=dict(color="#ff4ecd", width=3),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=430,
        title="Finish order comparison",
        yaxis_title="Finish position",
        xaxis_title="Driver",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def build_radar(race_df: pd.DataFrame, enriched_df: pd.DataFrame, selected_model: str):
    top3 = race_df.nsmallest(3, f"pred_{selected_model}")
    feature_cols = [
        "driver_recent_form_score",
        "track_specific_driver_score",
        "street_circuit_score",
        "quali_to_race_conversion",
        "constructor_recent_form_score",
    ]
    labels = ["Recent form", "Track fit", "Street fit", "Race craft", "Team form"]
    fig = go.Figure()
    enriched_lookup = enriched_df[enriched_df["race_id"] == race_df["race_id"].iloc[0]].set_index("driver") if not enriched_df.empty else pd.DataFrame()
    for _, row in top3.iterrows():
        if row["driver"] not in getattr(enriched_lookup, "index", []):
            continue
        values = enriched_lookup.loc[row["driver"], feature_cols].fillna(0).astype(float).to_list()
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=row["driver"],
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="rgba(0,0,0,0)"),
        height=430,
        title="Predicted podium driver strengths",
    )
    return fig


def render_image_gallery(title: str, paths: list[Path]):
    available = [path for path in paths if path.exists()]
    if not available:
        st.info(f"{title} will appear here after the pipeline generates those images.")
        return
    st.subheader(title)
    cols = st.columns(len(available))
    for col, path in zip(cols, available):
        col.image(str(path), use_container_width=True, caption=path.stem.replace("_", " ").title())


def main():
    inject_styles()

    metrics_report = load_json(REPORTS_DIR / "metrics_report.json", load_json(OUTPUT_DIR / "metrics_report.json", {}))
    predictions_df = load_dataframe(REPORTS_DIR / "test_predictions.csv")
    explanations_df = load_dataframe(REPORTS_DIR / "prediction_explanations.csv")
    enriched_df = load_dataframe(REPORTS_DIR / "enriched_raw_data.csv")
    ablation_df = load_dataframe(REPORTS_DIR / "ablation_report.csv")
    leakage_audit = load_json(REPORTS_DIR / "leakage_audit.json", {})
    run_summary = load_json(REPORTS_DIR / "run_summary.json", {})
    verify_report = load_json(VERIFY_DIR / "setup_report.json", {})
    run_manifest = get_latest_run_manifest()

    if predictions_df.empty or not metrics_report:
        st.warning("Run `python run.py --skip-collect` first so the dashboard can load fresh artifacts.")
        return

    available_models = [column.replace("pred_", "") for column in predictions_df.columns if column.startswith("pred_")]
    default_model = DEFAULT_DASHBOARD_MODEL if DEFAULT_DASHBOARD_MODEL in available_models else available_models[0]

    with st.sidebar:
        st.markdown("## Control Deck")
        selected_race = st.selectbox("Race", options=sorted(predictions_df["race_id"].unique()), index=len(sorted(predictions_df["race_id"].unique())) - 1)
        selected_model = st.selectbox("Prediction model", options=available_models, index=available_models.index(default_model))
        st.markdown("### Run manifest")
        if run_manifest:
            st.json(run_manifest)
        else:
            st.caption("No run manifest yet.")
        st.markdown("### Model limitations")
        st.caption("Telemetry visuals are explainability artifacts, not pre-race features. Probability heads are calibrated but still data-limited on rare events.")

    race_df = predictions_df[predictions_df["race_id"] == selected_race].copy().sort_values(f"pred_{selected_model}")

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    render_hero(metrics_report, predictions_df, race_df, selected_race, selected_model)
    st.markdown("</div>", unsafe_allow_html=True)
    render_status_band(run_manifest, run_summary, leakage_audit, verify_report)
    render_orientation_panel()
    render_prediction_workflow()
    render_selected_race_summary(race_df, selected_race, selected_model)

    tab_overview, tab_models, tab_race, tab_telemetry, tab_explanations = st.tabs(
        ["Overview", "Model Lab", "Race Pulse", "Telemetry Bay", "Explanations"]
    )

    with tab_overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Drivers in race", len(race_df))
        c2.metric("Best podium chance", f"{race_df['podium_probability'].max():.1%}")
        c3.metric("Highest DNF risk", f"{race_df['dnf_probability'].max():.1%}")
        c4.metric("Points lock", f"{race_df['points_probability'].max():.1%}")
        st.plotly_chart(build_model_bar(metrics_report), use_container_width=True)
        if leakage_audit:
            st.caption(
                f"Leakage audit: {leakage_audit.get('status', 'unknown')} | rows audited: {leakage_audit.get('rows_audited', 'n/a')}"
            )
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.plotly_chart(build_finish_order_chart(race_df, selected_model), use_container_width=True)
        with col_b:
            st.plotly_chart(build_probability_chart(race_df), use_container_width=True)

    with tab_models:
        leaderboard = pd.DataFrame(metrics_report).T.reset_index().rename(columns={"index": "model"})
        leaderboard = leaderboard.sort_values("mae")
        st.dataframe(leaderboard, use_container_width=True)
        if not ablation_df.empty:
            st.plotly_chart(build_ablation_chart(ablation_df), use_container_width=True)
        if not enriched_df.empty:
            st.plotly_chart(build_radar(race_df, enriched_df, selected_model), use_container_width=True)
        scatter = px.scatter(
            predictions_df,
            x="actual_finish",
            y=f"pred_{selected_model}",
            color="podium_probability",
            hover_data=["driver", "race_id", "dnf_probability", "points_probability"],
            template="plotly_dark",
            color_continuous_scale=["#09121f", "#00f5d4", "#ff4ecd"],
            title=f"Prediction spread for {selected_model}",
        )
        scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(scatter, use_container_width=True)

    with tab_race:
        top3 = race_df.nsmallest(3, f"pred_{selected_model}")[
            ["driver", f"pred_{selected_model}", "podium_probability", "dnf_probability", "confidence_score"]
        ]
        st.markdown("### Predicted podium")
        st.dataframe(top3, use_container_width=True)
        pressure = race_df[
            ["driver", "actual_finish", f"pred_{selected_model}", "podium_probability", "points_probability", "dnf_probability", "confidence_score"]
        ]
        st.dataframe(pressure, use_container_width=True)

        if not enriched_df.empty:
            race_enriched = enriched_df[enriched_df["race_id"] == selected_race].copy()
            metrics = ["driver_recent_form_score", "constructor_recent_form_score", "quali_to_race_conversion", "dnf_risk_score"]
            if not race_enriched.empty:
                melted = race_enriched[["driver"] + metrics].melt("driver", var_name="feature", value_name="value")
                heatmap = px.imshow(
                    melted.pivot(index="driver", columns="feature", values="value").fillna(0),
                    aspect="auto",
                    color_continuous_scale=["#030712", "#00f5d4", "#ff4ecd"],
                    template="plotly_dark",
                    title="Race feature heatmap",
                )
                heatmap.update_layout(paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(heatmap, use_container_width=True)

        img_paths = [
            PLOTS_DIR / "latest_race_prediction_dashboard.png",
            PLOTS_DIR / "podium_probabilities_latest_race.png",
            PLOTS_DIR / "latest_race_rankblend_comparison.png",
        ]
        render_image_gallery("Race support visuals", img_paths)

    with tab_telemetry:
        st.markdown("<div class='micro-note'>Telemetry visuals are created after the explainer runs with cached FastF1 race sessions.</div>", unsafe_allow_html=True)
        telemetry_paths = sorted(TELEMETRY_DIR.glob(f"*{selected_race}*.png"))
        render_image_gallery("Telemetry and strategy outputs", telemetry_paths)
        if not telemetry_paths:
            render_image_gallery(
                "Saved explainer plots",
                [
                    PLOTS_DIR / "season_trend_driver_form.png",
                    PLOTS_DIR / "driver_strength_radar.png",
                ],
            )

    with tab_explanations:
        if explanations_df.empty:
            st.info("Per-prediction explanations will appear here after the XGBoost SHAP export runs.")
        else:
            race_explanations = explanations_df[explanations_df["race_id"] == selected_race]
            st.dataframe(race_explanations, use_container_width=True)
        if leakage_audit:
            st.markdown("### Predictor guardrails")
            st.json(leakage_audit)
        render_image_gallery(
            "Explainability visuals",
            [
                OUTPUT_DIR / "shap_summary.png",
                OUTPUT_DIR / "shap_waterfall_example.png",
            ],
        )


if __name__ == "__main__":
    main()
