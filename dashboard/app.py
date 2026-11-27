"""InfraGuard AI dashboard (Streamlit). Talks to the FastAPI service only."""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
HEADERS = {"X-API-Key": os.environ["API_KEY"]} if os.getenv("API_KEY") else {}
SEV_COLOR = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#bcbd22"}

st.set_page_config(page_title="InfraGuard AI", page_icon="🛡️", layout="wide")


def api(path: str, method: str = "get", **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", headers=HEADERS, timeout=30, **kw)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API call {path} failed: {exc}")
        return None


def to_df(rows):
    df = pd.DataFrame(rows or [])
    for c in ("timestamp", "start", "end"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    return df


st.title("🛡️ InfraGuard AI")
st.caption("Simulated infrastructure telemetry. Detection is not root-cause analysis: hypotheses are hints for engineers.")

health = api("/health") or {}
c1, c2, c3 = st.columns(3)
c1.metric("API status", health.get("status", "unreachable"))
c2.metric("Active model", health.get("active_model", "-"))
c3.metric("Knowledge-base chunks", health.get("knowledge_base_chunks", 0))

with st.sidebar:
    st.header("Controls")
    refresh = st.selectbox("Auto-refresh", ["off", "10 s", "30 s"])
    window = st.slider("Chart window (samples)", 60, 1440, 360, step=60)
interval = {"off": None, "10 s": 10, "30 s": 30}[refresh]

tabs = st.tabs(["Overview", "Metrics", "Anomaly timeline", "Incidents", "Assistant", "Model performance"])


def overview():
    latest, incidents = to_df(api("/metrics")), to_df(api("/incidents", params={"limit": 200}))
    if latest.empty:
        st.info("No telemetry yet. Start the stream: `python scripts/stream_simulator.py`")
        return
    open_inc = incidents[incidents["status"] == "open"] if not incidents.empty else incidents
    k1, k2, k3 = st.columns(3)
    k1.metric("Servers reporting", latest["server_id"].nunique())
    k2.metric("Open incidents", len(open_inc))
    k3.metric("Total incidents", len(incidents))
    st.subheader("Server health")
    rows = []
    for r in latest.itertuples():
        sev = open_inc[open_inc["server_id"] == r.server_id]["severity"].tolist() if not open_inc.empty else []
        status = "HIGH" if "HIGH" in sev else "MEDIUM" if "MEDIUM" in sev else "LOW" if sev else "HEALTHY"
        rows.append({"server": r.server_id, "status": status, "cpu %": round(r.cpu_usage, 1),
                     "memory %": round(r.memory_usage, 1), "latency ms": round(r.latency, 1),
                     "packet loss %": round(r.packet_loss, 2), "last sample": r.timestamp})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def metrics():
    servers = sorted(to_df(api("/metrics"))["server_id"].unique()) if api("/metrics") else []
    if not servers:
        return
    sid = st.selectbox("Server", servers, key="metrics_server")
    df = to_df(api("/metrics", params={"server_id": sid, "limit": window}))
    if df.empty:
        return
    for title, cols in [("CPU & memory (%)", ["cpu_usage", "memory_usage"]),
                        ("Network (Mbps)", ["network_in", "network_out"]),
                        ("Latency (ms) & response time (ms)", ["latency", "response_time"]),
                        ("Packet loss (%)", ["packet_loss"])]:
        st.markdown(f"**{title}**")
        long = df.melt("timestamp", cols, var_name="metric", value_name="value")
        st.altair_chart(alt.Chart(long).mark_line().encode(x="timestamp:T", y="value:Q", color="metric:N")
                        .properties(height=180), width="stretch")


def timeline():
    an, inc = to_df(api("/anomalies", params={"limit": 3000})), to_df(api("/incidents", params={"limit": 500}))
    if an.empty:
        st.info("No anomalies recorded yet.")
        return
    st.markdown("**Flagged samples per server**")
    st.altair_chart(alt.Chart(an).mark_circle(size=40, opacity=0.6).encode(
        x="timestamp:T", y="server_id:N", color=alt.value("#d62728"),
        tooltip=["server_id", "timestamp", "score", "incident_id"]).properties(height=200),
        width="stretch")
    if not inc.empty:
        st.markdown("**Incident spans**")
        st.altair_chart(alt.Chart(inc).mark_bar(height=14).encode(
            x="start:T", x2="end:T", y="server_id:N",
            color=alt.Color("severity:N", scale=alt.Scale(domain=list(SEV_COLOR), range=list(SEV_COLOR.values()))),
            tooltip=["incident_id", "severity", "hypothesis", "start", "end"]).properties(height=200),
            width="stretch")


def incidents_tab():
    c1, c2 = st.columns(2)
    sev = c1.selectbox("Severity", ["all", "HIGH", "MEDIUM", "LOW"])
    sid = c2.text_input("Server (e.g. SERVER_02)", "")
    params = {"limit": 200, **({"severity": sev} if sev != "all" else {}), **({"server_id": sid} if sid else {})}
    df = to_df(api("/incidents", params=params))
    if df.empty:
        st.info("No incidents match.")
        return
    df["metrics"] = df["anomalous_metrics"].apply(", ".join)
    st.dataframe(df[["incident_id", "server_id", "severity", "status", "start", "end", "duration_min", "metrics",
                     "hypothesis", "recommended_action"]], hide_index=True, width="stretch")


with tabs[0]:
    (st.fragment(run_every=interval)(overview) if interval else overview)()
with tabs[1]:
    (st.fragment(run_every=interval)(metrics) if interval else metrics)()
with tabs[2]:
    (st.fragment(run_every=interval)(timeline) if interval else timeline)()
with tabs[3]:
    incidents_tab()

with tabs[4]:
    st.markdown("Ask about an incident, e.g. *Why is Server 02 experiencing high latency?*")
    q = st.text_input("Question", "Why is Server 02 experiencing high latency?")
    if st.button("Ask") and q:
        res = api("/assistant/query", "post", json={"question": q})
        if res:
            st.write(res["answer"])
            if res.get("low_confidence"):
                st.warning("Low retrieval confidence: documentation may not cover this.")
            if res.get("sources"):
                st.markdown("**Sources**")
                st.dataframe(pd.DataFrame(res["sources"]), hide_index=True)
            st.caption(f"answer mode: {res.get('mode')}")

with tabs[5]:
    rep = api("/model/report")
    if rep:
        cols = ["event_precision", "event_recall", "event_f1", "false_alerts_per_day_per_server",
                "median_detection_latency_min", "pointwise_average_precision", "incident_grouping_ari",
                "alert_reduction_ratio"]
        table = pd.DataFrame({m: {c: r.get(c) for c in cols} for m, r in rep.items()}).T
        st.dataframe(table, width="stretch")
        st.markdown("**Recall by anomaly type**")
        st.dataframe(pd.DataFrame({m: r["recall_by_type"] for m, r in rep.items()}).T, width="stretch")
        st.caption("Event-based metrics on a held-out, chronologically later, labelled synthetic test set.")
