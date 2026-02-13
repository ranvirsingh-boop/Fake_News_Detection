import os
import requests


API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("GOOGLE_CSE_ID")

def google_search(query, num_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": num_results
    }
    response = requests.get(url, params=params)
    return response.json()

def extract_evidence(search_data):
    evidence = []

    if "items" not in search_data:
        return evidence

    for item in search_data["items"]:
        evidence.append({
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", "")
        })

    return evidence

def verify_claim(claim, evidence_list):
    claim_lower = claim.lower()

    support = 0
    contradict = 0

    for e in evidence_list:
        text = (e["title"] + " " + e["snippet"]).lower()

        if any(word in text for word in claim_lower.split()):
            support += 1
        if "not" in text or "false" in text or "misleading" in text:
            contradict += 1

    if contradict > support:
        return "Contradicted by sources"
    elif support > 0:
        return "Supported by sources"
    else:
        return "No reliable source found"
