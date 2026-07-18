"""Diagnostic: shows exactly what OPENAI_API_KEY value gets loaded from
.env, without printing the secret itself -- just its length and
first/last few characters, plus whether it has hidden whitespace or
stray quote characters. Run this BEFORE the real smoke test if you're
getting an auth error despite having a working key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings

settings = get_settings()
key = settings.openai_api_key

print(f"Length of loaded key: {len(key)}")
print(f"First 8 chars: {key[:8]!r}")
print(f"Last 4 chars: {key[-4:]!r}")
print(f"Has leading/trailing whitespace: {key != key.strip()}")
print(f"Contains a quote character: {chr(34) in key or chr(39) in key}")
print(f"Contains a newline: {chr(10) in key or chr(13) in key}")

env_path = Path(".env")
print(f"\n.env file exists at: {env_path.resolve()}")
if env_path.exists():
    raw = env_path.read_bytes()
    print(f"First 3 bytes of .env (hex): {raw[:3].hex()}  (efbbbf = UTF-8 BOM, a common culprit)")
