import pytest
from geopulse.genai.analyst import summarize

class FakeResponse:
    def raise_for_status(self): pass
    def json(self): return {"output_text":"The evidence indicates elevated recurrence risk."}

def test_gemini_interactions_request_is_grounded(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY","test-secret")
    monkeypatch.setenv("GEMINI_MODEL","gemini-3.8-flash")
    captured={}
    def fake_post(url,**kwargs):
        captured["url"]=url; captured.update(kwargs); return FakeResponse()
    monkeypatch.setattr("geopulse.genai.analyst.requests.post",fake_post)
    answer=summarize("Which cell has the highest recurrence?",{"top_cell":"A12","probability":0.82})
    assert answer.startswith("The evidence")
    assert captured["url"].endswith("/v1beta/interactions")
    assert captured["headers"]["x-goog-api-key"]=="test-secret"
    assert captured["json"]["store"] is False
    assert "A12" in captured["json"]["input"]

def test_missing_gemini_key_is_explicit(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY",raising=False)
    with pytest.raises(RuntimeError): summarize("Explain risk",{"risk":0.5})
