"""Optional Gemini analytical summary grounded in caller-supplied computed evidence."""
import os
import requests

def summarize(question:str,evidence:dict)->str:
    api_key=os.getenv("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("GEMINI_API_KEY is not configured")
    if not question.strip() or len(question)>1000: raise ValueError("Question must contain 1-1000 characters")
    # Evidence is structured output from GeoPulse analytics, not arbitrary web context.
    prompt=("You are the GeoPulse analytical assistant. Answer only from the JSON evidence below. "
      "Do not invent locations, measurements, causes, inspections, or official engineering conclusions. "
      "If evidence is insufficient, say so. Distinguish recurrence probability from infrastructure severity. "
      "State uncertainty and refer to the evidence explicitly.\n\n"
      f"QUESTION:\n{question}\n\nEVIDENCE JSON:\n{evidence}")
    model=os.getenv("GEMINI_MODEL","gemini-2.0-flash")
    response=requests.post(
      f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
      params={"key":api_key},json={"contents":[{"parts":[{"text":prompt}]}],
        "generationConfig":{"temperature":0.2,"maxOutputTokens":700}},timeout=45)
    response.raise_for_status(); data=response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
