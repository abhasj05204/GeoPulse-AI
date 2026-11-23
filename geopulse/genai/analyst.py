"""Evidence-grounded Gemini summaries through the current Interactions REST API."""
import os
import requests

def summarize(question:str,evidence:dict)->str:
    api_key=os.getenv("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("GEMINI_API_KEY is not configured")
    if not question.strip() or len(question)>1000: raise ValueError("Question must contain 1-1000 characters")
    prompt=("You are the GeoPulse analytical assistant. Answer only from the JSON evidence below. "
      "Do not invent locations, measurements, causes, inspections, or official engineering conclusions. "
      "If evidence is insufficient, say so. Distinguish recurrence probability from infrastructure severity. "
      "State uncertainty and cite the exact evidence values used.\n\n"
      f"QUESTION:\n{question}\n\nEVIDENCE JSON:\n{evidence}")
    model=os.getenv("GEMINI_MODEL","gemini-3.8-flash")
    try:
        response=requests.post("https://generativelanguage.googleapis.com/v1beta/interactions",
          headers={"x-goog-api-key":api_key,"Content-Type":"application/json","Api-Revision":"2026-05-20"},
          json={"model":model,"input":prompt,"store":False},timeout=45)
        response.raise_for_status()
        output=response.json().get("output_text")
        if not output: raise RuntimeError("Gemini returned no text output")
        return output
    except requests.RequestException as exc:
        raise RuntimeError("Gemini request failed; verify API availability, model access, and credentials") from exc
