"""Streamlit izleme paneli: gecikme trendleri, uç nokta dağılımı, halüsinasyon skoru.

Çalıştırmak için: `streamlit run src/llmops/dashboard.py`
Veri kaynağı: src/serving/api.py'nin her istekte yazdığı metrics.sqlite3 (bkz. metrics_store.py).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.llmops.metrics_store import fetch_recent

st.set_page_config(page_title="TİHA-Asistan İzleme Paneli", layout="wide")
st.title("TİHA-Asistan — LLMOps İzleme Paneli")

rows = fetch_recent(limit=500)
if not rows:
    st.info("Henüz kaydedilmiş istek yok. Önce FastAPI servisini (src/serving/api.py) çalıştırıp birkaç istek gönderin.")
    st.stop()

df = pd.DataFrame(rows)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

col1, col2, col3 = st.columns(3)
col1.metric("Toplam istek", len(df))
col2.metric("Ortalama gecikme (ms)", f"{df['latency_ms'].mean():.1f}")
hallucination_rate = (df["hallucination_score"].dropna() > 0.5).mean() if df["hallucination_score"].notna().any() else 0.0
col3.metric("Halüsinasyon oranı (>0.5)", f"{hallucination_rate:.1%}")

st.subheader("Zaman içinde gecikme")
st.line_chart(df.set_index("timestamp")["latency_ms"])

st.subheader("Uç nokta başına istek sayısı")
st.bar_chart(df["endpoint"].value_counts())

if df["hallucination_score"].notna().any():
    st.subheader("Halüsinasyon skoru dağılımı")
    st.bar_chart(df["hallucination_score"].dropna())

st.subheader("Model versiyonu karşılaştırması")
if df["model_version"].notna().any():
    version_stats = df.groupby("model_version")["latency_ms"].agg(["mean", "count"])
    st.dataframe(version_stats)

st.subheader("Son istekler")
st.dataframe(df[["timestamp", "endpoint", "model_version", "latency_ms", "hallucination_score"]].head(50))
