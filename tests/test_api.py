from fastapi.testclient import TestClient
import backend.main as api
client=TestClient(api.app)

def test_health():
    response=client.get("/api/health")
    assert response.status_code==200 and response.json()["status"]=="ok"

def test_hotspot_endpoint_clusters_coordinates():
    points=[{"latitude":40.7,"longitude":-73.9},{"latitude":40.7001,"longitude":-73.9001},{"latitude":40.8,"longitude":-73.8}]
    response=client.post("/api/hotspots",json={"points":points,"eps_m":100,"min_samples":2})
    assert response.status_code==200
    assert response.json()["cluster_count"]==1

def test_ai_ask_passes_current_evidence(monkeypatch):
    risk_data={"as_of":"2025-01-01","cells":[
      {"cx":1,"cy":2,"latitude":40.7,"longitude":-73.9,"risk_level":"High","recurrence_probability":0.8},
      {"cx":3,"cy":4,"latitude":40.8,"longitude":-73.8,"risk_level":"Low","recurrence_probability":0.1}]}
    monkeypatch.setattr(api,"latest_risks",lambda:risk_data)
    monkeypatch.setattr(api,"performance",lambda:{"best_model":"random_forest"})
    monkeypatch.setattr(api,"explain",lambda:{"features":[{"feature":"count_30d","importance_mean":0.2}]})
    captured={}
    def fake_summarize(question,evidence):
        captured.update(evidence); return "Summary grounded in the supplied evidence."
    monkeypatch.setattr(api,"summarize",fake_summarize)
    response=client.post("/api/ai/ask",json={"question":"Which cell is highest risk?"})
    assert response.status_code==200
    assert captured["top_10_recurrence_cells"][0]["cx"]==1
    assert captured["risk_band_counts"]["High"]==1
