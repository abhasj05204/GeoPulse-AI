"""Small HTTP client shared by the Streamlit dashboard and its tests."""
import os
import requests

def api_base_url(value:str|None=None)->str:
    base=(value or os.getenv("GEOPULSE_API_URL") or "http://127.0.0.1:8000").strip().rstrip("/")
    return base if "://" in base else f"http://{base}"

def get_json(base:str,path:str,timeout:float=20):
    response=requests.get(f"{base.rstrip('/')}{path}",timeout=timeout)
    response.raise_for_status()
    return response.json()

def post_json(base:str,path:str,payload:dict,timeout:float=30):
    response=requests.post(f"{base.rstrip('/')}{path}",json=payload,timeout=timeout)
    response.raise_for_status()
    return response.json()
