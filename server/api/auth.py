import os
from dotenv import load_dotenv
from pathlib import Path
from rest_framework.permissions import BasePermission

# Load env (server/.env reading is already set in settings, but safe)
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


class HasAPIKey(BasePermission):
    """
    Require header: X-API-Key: <key>
    """
    def has_permission(self, request, view):
        expected = os.getenv("DOCQA_API_KEY", "")
        if not expected:
            # If no key set, allow (dev mode). For strict prod, return False instead.
            return True
        provided = request.headers.get("X-API-Key", "")
        return provided == expected
