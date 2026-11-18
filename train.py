from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
DATA = BASE / "data" / "crop_yield.csv"
OUT = BASE / "model" / "artifacts.joblib"
PERCENTILE = 99


def build_features(area, production, typical):
    y = production / area
    return np.column_stack([np.log1p(area), np.log(y), np.log(y / typical)])


def main():
    df = pd.read_csv(DATA).dropna(subset=["production"])
    df = df[(df["area"] > 0) & (df["production"] > 0)].copy()
    df["yield"] = df["production"] / df["area"]
    df["key"] = df["state_name"] + "|" + df["crop_name"]
    df["year_start"] = df["year"].str[:4].astype(int)

    typical = df.groupby("key")["yield"].median()
    df["typical"] = df["key"].map(typical)

    X = build_features(df["area"].to_numpy(), df["production"].to_numpy(), df["typical"].to_numpy())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    sample = Xs[np.random.RandomState(42).choice(len(Xs), 10000, replace=False)]
    scores = {}
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(sample)
        scores[k] = silhouette_score(sample, labels)
    best_k = max(scores, key=scores.get)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xs)
    dist = km.transform(Xs).min(axis=1)
    threshold = float(np.percentile(dist, PERCENTILE))
    df["flag"] = dist > threshold

    options = {s: sorted(g["crop_name"].unique().tolist()) for s, g in df.groupby("state_name")}
    trend = {}
    for k, g in df.groupby("key"):
        t = g.groupby("year_start")["yield"].mean().sort_index()
        trend[k] = {"years": t.index.tolist(), "avg": t.round(4).tolist()}
    group = df.groupby("key").agg(area=("area", "median"), typical=("yield", "median"))
    stats = {k: {"area": float(r["area"]), "typical": float(r["typical"])} for k, r in group.iterrows()}

    meta = {
        "records": int(len(df)),
        "anomalies": int(df["flag"].sum()),
        "k": int(best_k),
        "silhouette": float(scores[best_k]),
        "threshold": threshold,
        "percentile": PERCENTILE,
    }

    OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "km": km, "options": options, "trend": trend, "stats": stats, "meta": meta}, OUT)
    return meta, scores


if __name__ == "__main__":
    m, s = main()
    for k, v in s.items():
        print(f"k = {k}  silhouette = {v:.3f}")
    print(f"Chosen k            : {m['k']}")
    print(f"Records used        : {m['records']}")
    print(f"Distance threshold  : {m['threshold']:.3f} (top {100 - m['percentile']}% farthest records)")
    print(f"Anomalies flagged   : {m['anomalies']} ({m['anomalies'] / m['records'] * 100:.2f}%)")
    print(f"Saved to            : {OUT}")
