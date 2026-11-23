"""Streamlit analytical dashboard for GeoPulse model and spatial API results."""
import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
st.set_page_config(page_title="GeoPulse AI",page_icon="🗺️",layout="wide")
st.title("GeoPulse AI"); st.caption("Predict. Prioritize. Prevent. · Geospatial complaint recurrence intelligence")
api=os.getenv("GEOPULSE_API_URL","http://127.0.0.1:8000").rstrip("/")
def get(path):
    r=requests.get(f"{api}{path}",timeout=10); r.raise_for_status(); return r.json()
try:
    health=get("/api/health"); a,b=st.columns(2); a.metric("API",health["status"]); b.metric("Trained model","Available" if health["model_available"] else "Not trained")
except requests.RequestException: st.error("FastAPI is unavailable. Start the backend to load live analytics."); st.stop()
tabs=st.tabs(["Overview & model","Hotspot explorer","Prediction","AI analyst"])
with tabs[0]:
    try:
        report=get("/api/performance"); st.subheader("Chronological holdout evaluation"); st.caption(f"Selected model: {report['best_model']} · cutoff: {report['temporal_cutoff']}")
        metrics=pd.DataFrame(report["metrics"]).T; st.dataframe(metrics, use_container_width=True)
        explain=get("/api/explain"); imp=pd.DataFrame(explain["features"]); st.subheader("Permutation feature importance"); st.plotly_chart(px.bar(imp.sort_values("importance_mean"),x="importance_mean",y="feature",orientation="h",error_x="importance_std"),use_container_width=True)
    except requests.RequestException as exc: st.info(f"Train a model to populate evaluation and explainability reports. ({exc})")
with tabs[1]:
    st.subheader("Explore complaint locations"); upload=st.file_uploader("Upload a CSV with latitude and longitude",type=["csv"],key="geo")
    if upload:
        data=pd.read_csv(upload)
        if not {"latitude","longitude"}.issubset(data): st.error("CSV must contain latitude and longitude columns.")
        else:
            points=data.dropna(subset=["latitude","longitude"]).head(3000).to_dict(orient="records")
            try:
                r=requests.post(f"{api}/api/hotspots",json={"points":points,"eps_m":500,"min_samples":8},timeout=60); r.raise_for_status(); result=r.json(); spatial=pd.DataFrame(result["points"])
                st.write(f"Clusters: {result['cluster_count']} · Noise points: {result['noise_count']} · Mapped sample: {len(spatial)}")
                fig=px.scatter_mapbox(spatial,lat="latitude",lon="longitude",color=spatial["cluster"].astype(str),hover_data=[c for c in ["complaint_type","cluster"] if c in spatial],zoom=10,mapbox_style="open-street-map",height=600)
                st.plotly_chart(fig,use_container_width=True)
            except requests.RequestException as exc: st.error(f"Hotspot analysis failed: {exc}")
with tabs[2]:
    st.subheader("Cell-level recurrence estimate"); st.caption("Use engineered features from the training pipeline; sample defaults are illustrative inputs.")
    defaults={"count_7d":2.0,"count_30d":8.0,"count_90d":18.0,"mean_daily_30d":.267,"days_since_last":4.0,"day_of_week":1.0,"month":6.0,"x_km":580.0,"y_km":4500.0}
    with st.form("prediction"):
        cols=st.columns(3); features={}
        for i,(name,value) in enumerate(defaults.items()): features[name]=cols[i%3].number_input(name,value=float(value))
        submitted=st.form_submit_button("Estimate recurrence risk")
    if submitted:
        try:
            r=requests.post(f"{api}/api/predict",json={"features":features},timeout=30); r.raise_for_status(); out=r.json()
            st.metric("30-day recurrence probability",f"{out['recurrence_probability']:.1%}"); st.metric("Risk band",out["risk_level"]); st.json(out)
        except requests.RequestException as exc: st.error(str(exc))
with tabs[3]:
    st.subheader("Evidence-grounded AI analyst"); question=st.text_input("Ask about a model or spatial result")
    evidence_text=st.text_area("Analytical evidence (JSON)",value='{"model_status":"not loaded","note":"Paste API evaluation, hotspot, or prediction output here."}',height=150)
    if st.button("Generate evidence-based summary"):
        import json
        try:
            r=requests.post(f"{api}/api/ai/summary",json={"question":question,"evidence":json.loads(evidence_text)},timeout=60); r.raise_for_status(); st.write(r.json()["summary"])
        except (ValueError,requests.RequestException) as exc: st.error(f"Could not generate summary: {exc}")
