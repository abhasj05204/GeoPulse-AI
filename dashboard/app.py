"""Streamlit analytical dashboard consuming GeoPulse API results."""
import os,json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
st.set_page_config(page_title="GeoPulse AI",page_icon="🗺️",layout="wide")
st.title("GeoPulse AI"); st.caption("Predict. Prioritize. Prevent. · Geospatial complaint recurrence intelligence")
api=os.getenv("GEOPULSE_API_URL","http://127.0.0.1:8000").rstrip("/")
if "://" not in api: api=f"http://{api}"
def get(path):
    r=requests.get(f"{api}{path}",timeout=20); r.raise_for_status(); return r.json()
try:
    health=get("/api/health"); a,b=st.columns(2); a.metric("API",health["status"]); b.metric("Trained model","Available" if health["model_available"] else "Not trained")
except requests.RequestException: st.error("FastAPI is unavailable. Start the backend to load live analytics."); st.stop()
tabs=st.tabs(["Overview & model","Risk map","Hotspot explorer","Prediction","AI analyst"])
with tabs[0]:
    try:
        summary=get("/api/analytics/summary"); cards=st.columns(4)
        cards[0].metric("Panel rows",f"{summary['panel_rows']:,}"); cards[1].metric("Spatial cells",f"{summary['spatial_cells']:,}")
        cards[2].metric("Complaint events",f"{summary['complaint_events']:,}"); cards[3].metric("Observed through",summary["date_end"])
        report=get("/api/performance"); st.subheader("Chronological holdout evaluation")
        st.caption(f"Selected model: {report['best_model']} · cutoff: {report['temporal_cutoff']}")
        st.dataframe(pd.DataFrame(report["metrics"]).T,use_container_width=True)
        importance=get("/api/explain"); imp=pd.DataFrame(importance["features"]); st.subheader("Permutation feature importance")
        st.plotly_chart(px.bar(imp.sort_values("importance_mean"),x="importance_mean",y="feature",orientation="h",error_x="importance_std"),use_container_width=True)
    except requests.RequestException as exc: st.info(f"Run the data and model pipeline to populate this view. ({exc})")
with tabs[1]:
    st.subheader("Latest cell-level recurrence risk")
    try:
        risks=get("/api/risks"); risk=pd.DataFrame(risks["cells"])
        st.caption(f"Latest observed date: {risks['as_of']} · Recurrence probability is not defect severity.")
        if not risk.empty:
            fig=px.scatter_mapbox(risk,lat="latitude",lon="longitude",color="risk_level",size="risk_score",
              hover_data=["recurrence_probability","risk_score","cx","cy"],color_discrete_map={"High":"#d62728","Medium":"#ff7f0e","Low":"#2ca02c"},
              zoom=10,mapbox_style="open-street-map",height=650)
            st.plotly_chart(fig,use_container_width=True)
    except requests.RequestException as exc: st.info(f"Risk map unavailable until data and model artifacts are generated. ({exc})")
with tabs[2]:
    st.subheader("Explore complaint locations with DBSCAN"); upload=st.file_uploader("Upload CSV with latitude and longitude",type=["csv"],key="geo")
    if upload:
        data=pd.read_csv(upload)
        if not {"latitude","longitude"}.issubset(data): st.error("CSV must contain latitude and longitude columns.")
        else:
            points=data.dropna(subset=["latitude","longitude"]).head(3000).to_dict(orient="records")
            try:
                r=requests.post(f"{api}/api/hotspots",json={"points":points,"eps_m":500,"min_samples":8},timeout=60); r.raise_for_status()
                result=r.json(); spatial=pd.DataFrame(result["points"]); st.write(f"Clusters: {result['cluster_count']} · Noise points: {result['noise_count']}")
                fig=px.scatter_mapbox(spatial,lat="latitude",lon="longitude",color=spatial["cluster"].astype(str),zoom=10,mapbox_style="open-street-map",height=600)
                st.plotly_chart(fig,use_container_width=True)
            except requests.RequestException as exc: st.error(f"Hotspot analysis failed: {exc}")
with tabs[3]:
    st.subheader("Cell-level recurrence estimate"); st.caption("Sample values are illustrative inputs; use features from the training pipeline.")
    defaults={"count_7d":2.0,"count_30d":8.0,"count_90d":18.0,"mean_daily_30d":.267,"days_since_last":4.0,"neighbor_count_30d":12.0,"day_of_week":1.0,"month":6.0,"x_km":580.0,"y_km":4500.0}
    with st.form("prediction"):
        cols=st.columns(3); features={}
        for i,(name,value) in enumerate(defaults.items()): features[name]=cols[i%3].number_input(name,value=float(value))
        submitted=st.form_submit_button("Estimate recurrence risk")
    if submitted:
        try:
            r=requests.post(f"{api}/api/predict",json={"features":features},timeout=30); r.raise_for_status(); out=r.json()
            st.metric("30-day recurrence probability",f"{out['recurrence_probability']:.1%}"); st.metric("Risk band",out["risk_level"]); st.json(out)
        except requests.RequestException as exc: st.error(str(exc))
with tabs[4]:
    st.subheader("GeoPulse AI analyst"); question=st.text_input("Ask about current risk zones, trends, or model performance")
    if st.button("Ask GeoPulse"):
        if not question.strip(): st.warning("Enter a question first.")
        else:
            try:
                r=requests.post(f"{api}/api/ai/ask",json={"question":question},timeout=60); r.raise_for_status()
                answer=r.json(); st.write(answer["summary"])
                with st.expander("View evidence supplied to the model"): st.json(answer["evidence_used"])
            except requests.RequestException as exc: st.error(f"AI analyst unavailable: {exc}")
