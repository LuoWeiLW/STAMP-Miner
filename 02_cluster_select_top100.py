# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from core.peptide_features import feature_vector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_xlsx", required=True)
    ap.add_argument("--out_xlsx", required=True)
    ap.add_argument("--top_n", type=int, default=100)
    ap.add_argument("--n_clusters", type=int, default=20)
    ap.add_argument("--method", default="ward", choices=["ward","average","complete"])
    args = ap.parse_args()

    df = pd.read_excel(Path(args.ranked_xlsx))
    df = df.sort_values("Score_total", ascending=False).reset_index(drop=True)
    df_top = df.head(max(args.top_n, args.n_clusters)).copy()

    X = np.vstack([feature_vector(s, normalize=True) for s in df_top["sequence"].astype(str).values])
    Xs = StandardScaler().fit_transform(X)

    if args.method == "ward":
        labels = AgglomerativeClustering(n_clusters=int(args.n_clusters), linkage="ward").fit_predict(Xs)
    else:
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        labels = AgglomerativeClustering(n_clusters=int(args.n_clusters), linkage=args.method).fit_predict(Xn)

    df_top["cluster"] = labels.astype(int)

    selected = []
    for cl in sorted(df_top["cluster"].unique().tolist()):
        selected.append(df_top[df_top["cluster"]==cl].sort_values("Score_total", ascending=False).iloc[0])
    sel_df = pd.DataFrame(selected).sort_values("Score_total", ascending=False).head(args.top_n)

    if len(sel_df) < args.top_n:
        used = set(sel_df["sequence"].tolist())
        fill = df_top[~df_top["sequence"].isin(used)].sort_values("Score_total", ascending=False).head(args.top_n - len(sel_df))
        sel_df = pd.concat([sel_df, fill], ignore_index=True).sort_values("Score_total", ascending=False).head(args.top_n)

    out = Path(args.out_xlsx); out.parent.mkdir(parents=True, exist_ok=True)
    sel_df.to_excel(out, index=False)
    print(f"[OK] Saved Top{args.top_n} cluster-selected table: {out}")

if __name__ == "__main__":
    main()
