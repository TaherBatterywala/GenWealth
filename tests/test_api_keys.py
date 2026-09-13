"""
GenWealth — API Key Health & Diagnostics Suite
==============================================
File: tests/test_api_keys.py

This script independently tests every API key configured in `.env`:
  1. Groq API Keys (GROQ_API_KEY, GROQ_API_KEY2, ...)
  2. Google Gemini API Keys (GEMINI_API_KEY, GEMINI_API_KEY2, ... GEMINI_API_KEY6)
  3. MongoDB URI (MONGODB_URI)

Updated to test the latest Google Gemini / Gemma model catalog:
  - gemini-3.1-flash-lite (Recommended: 15 RPM / 500 RPD free tier)
  - gemini-3.7-flash
  - gemini-3.6-flash
  - gemini-3.5-flash
  - gemini-2.5-flash-lite
  - gemma-4-26b / gemma-4-31b
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

print("=" * 80)
print("  GENWEALTH API KEY DIAGNOSTICS & VERIFICATION TOOL")
print(f"  Loaded .env from: {ENV_PATH}")
print("=" * 80)
print()

# ============================================================================
# 1. GROQ API KEYS TESTING
# ============================================================================
print("=" * 80)
print("1. GROQ API KEYS DIAGNOSTICS")
print("=" * 80)

groq_keys = {
    k: v.strip()
    for k, v in os.environ.items()
    if k.startswith("GROQ_API_KEY") and v.strip()
}

if not groq_keys:
    print("  [ERROR] No GROQ_API_KEY found in .env")
else:
    try:
        from groq import Groq
        for name, key in groq_keys.items():
            masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            print(f"\n--- Testing {name} ({masked_key}) ---")
            
            # Format check
            if not key.startswith("gsk_"):
                print(f"  [WARN] Key does not start with standard prefix 'gsk_'. Current prefix: {key[:4]}")
            else:
                print("  [INFO] Key format prefix valid ('gsk_')")
                
            client = Groq(api_key=key)
            
            # 1. List models
            try:
                models_resp = client.models.list()
                model_ids = sorted([m.id for m in models_resp.data])
                print(f"  [SUCCESS] API Authentication OK! Found {len(model_ids)} accessible models:")
                for m_id in model_ids:
                    print(f"    - {m_id}")
            except Exception as e:
                print(f"  [FAIL] Failed to list models: {e}")
                continue

            # 2. Test generation on primary & fallback models
            test_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
            for model_name in test_models:
                try:
                    chat_resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": "Respond with the single word: OK"}],
                        max_tokens=30,
                    )
                    content = (chat_resp.choices[0].message.content or "").strip()
                    print(f"  [OK] Model '{model_name}': Responded with -> {repr(content[:60])}")
                except Exception as e:
                    err_str = str(e)
                    if "model_not_found" in err_str or "does not exist" in err_str:
                        print(f"  [MODEL UNAVAILABLE] Model '{model_name}': Not enabled/available for this account tier.")
                    else:
                        print(f"  [FAIL] Model '{model_name}': Error: {err_str[:120]}")

    except ImportError:
        print("  [FATAL] 'groq' package is not installed. Run: pip install groq")

# ============================================================================
# 2. GOOGLE GEMINI API KEYS TESTING
# ============================================================================
print("\n" + "=" * 80)
print("2. GOOGLE GEMINI / GEMMA NEW MODEL CATALOG TESTING")
print("=" * 80)

gemini_keys = {
    k: v.strip()
    for k, v in os.environ.items()
    if k.startswith("GEMINI_API_KEY") and v.strip()
}

# The user's requested new models to test:
CANDIDATE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemma-4-26b"
]

if not gemini_keys:
    print("  [ERROR] No GEMINI_API_KEY found in .env")
else:
    for name, key in gemini_keys.items():
        masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
        print(f"\n--- Testing {name} ({masked_key}) ---")
        
        # 1. Format check
        if key.startswith("AIzaSy"):
            print("  [INFO] Key format: Standard Google AI Studio (prefix 'AIzaSy')")
        elif key.startswith("AQ."):
            print("  [INFO] Key format: GCP / Cloud Code key (prefix 'AQ.')")
        else:
            print(f"  [INFO] Key prefix: {key[:6]}...")

        # 2. Test each candidate model via REST generateContent
        working_models_for_key = []
        for model_name in CANDIDATE_MODELS:
            clean_m = model_name if model_name.startswith("models/") else f"models/{model_name}"
            gen_url = f"https://generativelanguage.googleapis.com/v1beta/{clean_m}:generateContent?key={key}"
            try:
                res = requests.post(
                    gen_url,
                    json={"contents": [{"parts": [{"text": "Reply with exactly the word: OK"}]}]},
                    timeout=8,
                )
                if res.status_code == 200:
                    resp_json = res.json()
                    candidates = resp_json.get("candidates", [])
                    if candidates:
                        part_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                        print(f"  [OK - 200] Model '{model_name}': Responded -> {repr(part_text[:50])}")
                        working_models_for_key.append(model_name)
                    else:
                        print(f"  [OK - 200] Model '{model_name}': Empty candidate payload")
                elif res.status_code == 404:
                    err_msg = res.json().get("error", {}).get("message", "Not found")
                    print(f"  [FAIL - 404] Model '{model_name}': {err_msg[:80]}")
                elif res.status_code == 429:
                    print(f"  [RATE LIMIT - 429] Model '{model_name}': Quota exceeded on this key.")
                else:
                    err_msg = res.json().get("error", {}).get("message", res.text[:80])
                    print(f"  [FAIL - {res.status_code}] Model '{model_name}': {err_msg[:80]}")
            except requests.Timeout:
                print(f"  [TIMEOUT] Model '{model_name}': Request timed out after 8s")
            except Exception as e:
                print(f"  [ERROR] Model '{model_name}': {e}")

        # 3. Test with Google GenAI SDK using best working model
        if working_models_for_key:
            best_model = working_models_for_key[0]
            try:
                from google import genai
                from google.genai import types as gt
                client = genai.Client(api_key=key)
                sdk_m = best_model if best_model.startswith("models/") else f"models/{best_model}"
                resp = client.models.generate_content(
                    model=sdk_m,
                    contents="Respond with the single word: OK",
                    config=gt.GenerateContentConfig(max_output_tokens=20),
                )
                text = (resp.text or "").strip()
                print(f"  [SDK VERIFIED] Google GenAI SDK successfully connected with '{sdk_m}': -> {repr(text[:40])}")
            except Exception as sdk_err:
                print(f"  [SDK NOTE] SDK test on '{best_model}': {str(sdk_err)[:100]}")
        else:
            print(f"  [SUMMARY] No candidate models were reachable with {name}.")

# ============================================================================
# 3. MONGODB DATABASE TESTING
# ============================================================================
print("\n" + "=" * 80)
print("3. MONGODB ATLAS CONNECTION DIAGNOSTICS")
print("=" * 80)

mongo_uri = os.environ.get("MONGODB_URI", "").strip()
if not mongo_uri:
    print("  [WARN] MONGODB_URI not found in .env")
else:
    # Mask password for display
    masked_uri = mongo_uri
    if "@" in mongo_uri and "://" in mongo_uri:
        prefix, rest = mongo_uri.split("://", 1)
        creds, host = rest.split("@", 1)
        if ":" in creds:
            user, _ = creds.split(":", 1)
            masked_uri = f"{prefix}://{user}:****@{host}"
    print(f"  URI: {masked_uri}")
    
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi
        
        client = MongoClient(mongo_uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("  [SUCCESS] Pinged your MongoDB Atlas deployment. Connection OK!")
        db_names = client.list_database_names()
        print(f"  [INFO] Accessible Databases ({len(db_names)}): {', '.join(db_names)}")
    except ImportError:
        print("  [WARN] 'pymongo' not installed. Run: pip install pymongo")
    except Exception as e:
        print(f"  [FAIL] MongoDB connection failed: {e}")

print("\n" + "=" * 80)
print("  DIAGNOSTICS COMPLETE")
print("=" * 80)
