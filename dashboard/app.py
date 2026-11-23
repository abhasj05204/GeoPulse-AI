"""Streamlit analytical dashboard consuming GeoPulse API results."""
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.api_client import api_base_url,get_json,post_json
load_dotenv()
st.set_page_config(page_title="GeoPulse AI",page_icon="🗺️",layout="wide")
st.title("GeoPulse AI"); st.caption("Predict. Prioritize. Prevent. · Geospatial complaint recurrence intelligence")
api=api_base_url()
def get(path,timeout=20): return get_json(api,path,timeout)
def post(path,payload,timeout=30): return post_json(api,path,payload,timeout)
try:
    health=get("/api/health"); a,b=st.columns(2); a.metric("API",health["status"]); b.metric("Trained model","Available" if health["model_available"] else "Not trained")
except requests.RequestException: st.error("FastAPI is unavailable. Start the backend to load live analytics."); st.stop()
tabs=st.tabs(["Overview & model","Risk map","Hotspot explorer","Prediction","AI analyst"])
with tabs[0]:
    try:
        summary=get("/api/analytics/summary"); cards=st.columns(4)
        cards[0].metric("Panel rows",f"{summary['panel_rows']:,}"); cards[1].metric("Spatial cells",f"{summary['spatial_cells']:,}")
        cards[2].metric("Events in labelled panel",f"{summary['panel_events']:,}"); cards[3].metric("Observed through",summary["date_end"])
        report=get("/api/performance"); model=get("/api/model")
        st.subheader("Model selection and holdout evaluation")
        st.caption(f"Selected by temporal-CV PR-AUC: {report['best_model']} · holdout starts {report['temporal_cutoff']}")
        st.dataframe(pd.DataFrame(report["metrics"]).T,use_container_width=True)
        base,threshold,holdout=st.columns(3)
        base.metric("Prior baseline PR-AUC",f"{report['baseline']['pr_auc']:.3f}")
        threshold.metric("Chosen operating threshold",f"{model['decision_threshold']:.3f}")
        holdout.metric("Selected model holdout PR-AUC",f"{report['metrics'][report['best_model']]['pr_auc']:.3f}")
        st.caption("Low/Medium/High are relative bands from temporal out-of-fold probabilities, not absolute severity levels.")
        trends=get("/api/analytics/trends"); trend=pd.DataFrame(trends["daily"])
        st.subheader("Daily complaint activity in the labelled panel")
        st.plotly_chart(px.line(trend,x="created_date",y="complaint_count",labels={"created_date":"Date","complaint_count":"Complaint count"}),use_container_width=True)
        importance=get("/api/explain"); imp=pd.DataFrame(importance["features"]); st.subheader("Permutation feature importance")
        st.plotly_chart(px.bar(imp.sort_values("importance_mean"),x="importance_mean",y="feature",orientation="h",error_x="importance_std"),use_container_width=True)
    except requests.RequestException as exc: st.info(f"Run the data and model pipeline to populate this view. ({exc})")
with tabs[1]:
    st.subheader("Latest cell-level recurrence risk")
    try:
        risks=get("/api/risks"); risk=pd.DataFrame(risks["cells"])
        st.caption(f"Latest observed date: {risks['as_of']} · Risk bands are cohort-relative; recurrence probability is not defect severity.")
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
                result=post("/api/hotspots",{"points":points,"eps_m":500,"min_samples":8},timeout=60)
                spatial=pd.DataFrame(result["points"]); st.write(f"Clusters: {result['cluster_count']} · Noise points: {result['noise_count']}")
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
            out=post("/api/predict",{"features":features})
            st.metric("30-day recurrence probability",f"{out['recurrence_probability']:.1%}"); st.metric("Risk band",out["risk_level"]); st.json(out)
        except requests.RequestException as exc: st.error(str(exc))
with tabs[4]:
    st.subheader("GeoPulse AI analyst"); question=st.text_input("Ask about current risk zones, trends, or model performance")
    if st.button("Ask GeoPulse"):
        if not question.strip(): st.warning("Enter a question first.")
        else:
            try:
                answer=post("/api/ai/ask",{"question":question},timeout=60); st.write(answer["summary"])
                with st.expander("View evidence supplied to the model"): st.json(answer["evidence_used"])
            except requests.RequestException as exc: st.error(f"AI analyst unavailable: {exc}")
