from pathlib import Path
import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import train

ART = Path(__file__).parent / "model" / "artifacts.joblib"
GREEN, RED, GREY = "#1f5c46", "#b3261e", "#8a9a93"

st.set_page_config(page_title="Crop Yield Anomaly Detection", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, [class*="st-"] { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
header[data-testid="stHeader"], footer, #MainMenu { display: none; }
div.block-container, .stMainBlockContainer { padding: 0 !important; max-width: 100% !important; }
.top { background: #16221d; color: #fff; padding: 18px 40px; display: flex; justify-content: space-between; align-items: baseline; }
.top .t { font-size: 22px; font-weight: 600; letter-spacing: .01em; }
.top .s { font-size: 13px; color: #b9c7c0; }
.st-key-body { max-width: 1240px; margin: 0 auto; padding: 28px 24px 48px 24px; }
.st-key-inputs, .st-key-result, .st-key-chart, .st-key-how { background: #fff; border: 1px solid #dde3e0; border-radius: 4px; padding: 22px 24px; }
.st-key-result { padding-bottom: 34px; }
.h { font-size: 15px; font-weight: 600; color: #16221d; margin: 0 0 14px 0; padding-bottom: 10px; border-bottom: 1px solid #e6ebe8; }
[data-testid="stWidgetLabel"] p { font-size: 13px; color: #4d5c55; font-weight: 500; }
[data-baseweb="select"] > div, [data-testid="stNumberInputContainer"] { border-radius: 3px !important; }
.calc { font-size: 13px; color: #4d5c55; margin-top: 4px; }
.calc b { color: #16221d; }
.verdict { padding: 12px 16px; border-left: 5px solid; margin-bottom: 16px; }
.verdict .v { font-size: 20px; font-weight: 600; }
.verdict .d { font-size: 13.5px; color: #3d4b44; margin-top: 3px; }
.v-bad { border-color: #b3261e; background: #fbefee; }
.v-bad .v { color: #b3261e; }
.v-ok { border-color: #1f5c46; background: #edf5f1; }
.v-ok .v { color: #1f5c46; }
.facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 18px; }
.fact { border: 1px solid #e3e8e5; padding: 10px 14px; border-radius: 3px; }
.fact span { display: block; font-size: 12px; color: #6a7973; }
.fact b { font-size: 19px; font-weight: 600; color: #16221d; }
.gauge-l { display: flex; justify-content: space-between; font-size: 12px; color: #6a7973; margin-bottom: 5px; }
.gauge { position: relative; height: 8px; background: #e6ebe8; border-radius: 2px; }
.gauge i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 2px; }
.gauge u { position: absolute; top: -4px; bottom: -4px; width: 2px; background: #16221d; }
.step { font-size: 13.5px; color: #34423b; line-height: 1.65; padding: 7px 0; border-bottom: 1px solid #eef1ef; }
.step:last-child { border-bottom: 0; }
.step b { color: #16221d; }
.hint { font-size: 12.5px; color: #6a7973; margin: 12px 0 0 0; }
@media (max-width: 900px) { .facts { grid-template-columns: 1fr; } .top { flex-direction: column; gap: 4px; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load():
    if not ART.exists():
        train.main()
    return joblib.load(ART)


with st.spinner("Preparing the model (first run only)"):
    A = load()

meta, options = A["meta"], A["options"]


def predict(state, crop, area, production):
    typical = A["stats"][f"{state}|{crop}"]["typical"]
    x = train.build_features(np.array([area]), np.array([production]), np.array([typical]))
    d = A["km"].transform(A["scaler"].transform(x))[0]
    return {"yield": production / area, "typical": typical, "dist": float(d.min()), "cluster": int(d.argmin()) + 1}


st.markdown(
    '<div class="top"><div class="t">Crop Yield Anomaly Detection</div>'
    '<div class="s">INT396 Unsupervised Learning &nbsp;|&nbsp; K-Means clustering on district-wise crop records</div></div>',
    unsafe_allow_html=True,
)

with st.container(key="body"):
    left, right = st.columns([1, 1.7], gap="large")

    with left:
        with st.container(key="inputs"):
            st.markdown('<div class="h">Crop record</div>', unsafe_allow_html=True)
            states = sorted(options)
            state = st.selectbox("State", states, index=states.index("Punjab") if "Punjab" in states else 0)
            crops = options[state]
            crop = st.selectbox("Crop", crops, index=crops.index("Rice") if "Rice" in crops else 0, key=f"crop_{state}")
            combo = f"{state}|{crop}"
            years = A["trend"][combo]["years"]
            if len(years) > 1:
                year = st.select_slider("Year", options=years, value=years[-1], format_func=lambda y: f"{y}-{str(y + 1)[2:]}", key=f"year_{combo}")
            else:
                year = years[0]
            s = A["stats"][combo]
            c1, c2 = st.columns(2)
            with c1:
                area = st.number_input("Area (hectares)", min_value=0.0, value=float(round(s["area"], 1)), step=100.0, format="%.1f", key=f"area_{combo}")
            with c2:
                production = st.number_input("Production (tonnes)", min_value=0.0, value=float(round(s["area"] * s["typical"], 1)), step=100.0, format="%.1f", key=f"prod_{combo}")
            shown = production / area if area > 0 else 0.0
            st.markdown(f'<div class="calc">Yield = production / area = <b>{shown:.2f} t/ha</b></div>', unsafe_allow_html=True)

    valid = area > 0 and production > 0
    r = predict(state, crop, area, production) if valid else None

    with right:
        with st.container(key="result"):
            st.markdown('<div class="h">Result</div>', unsafe_allow_html=True)
            if r is None:
                st.markdown('<p class="hint">Area and production must both be greater than zero.</p>', unsafe_allow_html=True)
            else:
                bad = r["dist"] > meta["threshold"]
                ratio = r["yield"] / r["typical"]
                rel = f"{ratio:.1f} times the typical yield" if ratio >= 1 else f"{ratio * 100:.0f}% of the typical yield"
                cls, label = ("v-bad", "Anomaly") if bad else ("v-ok", "Normal")
                text = f"Yield of {r['yield']:.2f} t/ha is {rel} for {crop} in {state}."
                pct = min(r["dist"] / (meta["threshold"] * 1.5), 1) * 100
                tick = 100 / 1.5
                st.markdown(
                    f'<div class="verdict {cls}"><div class="v">{label}</div><div class="d">{text}</div></div>'
                    '<div class="facts">'
                    f'<div class="fact"><span>Entered yield</span><b>{r["yield"]:.2f} t/ha</b></div>'
                    f'<div class="fact"><span>Typical yield</span><b>{r["typical"]:.2f} t/ha</b></div>'
                    f'<div class="fact"><span>Distance to nearest cluster</span><b>{r["dist"]:.2f}</b></div></div>'
                    f'<div class="gauge-l"><span>Close to a cluster centre</span><span>Limit {meta["threshold"]:.2f}</span></div>'
                    f'<div class="gauge"><i style="width:{pct:.1f}%;background:{RED if bad else GREEN}"></i><u style="left:{tick:.1f}%"></u></div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        with st.container(key="chart"):
            st.markdown(f'<div class="h">Yield trend: {crop} in {state}</div>', unsafe_allow_html=True)
            t = A["trend"][combo]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t["years"], y=t["avg"], name="Historical average", mode="lines", line=dict(color=GREY, width=2.2), fill="tozeroy", fillcolor="rgba(138,154,147,.16)"))
            if r is not None:
                base = t["avg"][t["years"].index(year)] if year in t["years"] else r["typical"]
                color = RED if r["dist"] > meta["threshold"] else GREEN
                fig.add_trace(go.Scatter(x=[year, year], y=[base, r["yield"]], name="Your entry", mode="lines+markers", line=dict(color=color, width=2, dash="dash"), marker=dict(color=color, size=[0, 13])))
            fig.update_layout(
                height=330, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=12, color="#4d5c55"), legend=dict(orientation="h", y=1.12, x=0),
            )
            fig.update_xaxes(title_text="Year", gridcolor="#eceff0", zeroline=False)
            fig.update_yaxes(title_text="Yield (tonnes per hectare)", gridcolor="#eceff0", zeroline=False)
            st.plotly_chart(fig, config={"displayModeBar": False}, key="trend_chart")

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    with st.container(key="how"):
        st.markdown('<div class="h">How the model decides</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="step"><b>Step 1.</b> Each record is turned into 3 numbers: log of area, log of yield, and log of yield compared with the typical yield of that crop in that state. These are standardised so all three carry equal weight.</div>'
            f'<div class="step"><b>Step 2.</b> K-Means groups the {meta["records"]:,} records into {meta["k"]} clusters (k = {meta["k"]} gave the best silhouette score, {meta["silhouette"]:.2f}).</div>'
            f'<div class="step"><b>Step 3.</b> For any record, the distance to its nearest cluster centre is measured. Records farther than {meta["threshold"]:.2f}, which is the farthest {100 - meta["percentile"]}% of the training data ({meta["anomalies"]:,} records), are called anomalies.</div>',
            unsafe_allow_html=True,
        )
