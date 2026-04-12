"""
Minimal health-check script used by Docker HEALTHCHECK.
Exits 0 if the /health endpoint returns HTTP 200, 1 otherwise.
Relies only on Python's stdlib — no extra packages required.
"""

import sys
import urllib.request

try:
    response = urllib.request.urlopen("http://localhost:8000/health", timeout=5)
    sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
