from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]  # Policy-Document-QA-Agent/
DEFAULT_INDEX_DIR = str(REPO_ROOT / "src" / ".index")  # keep index with your LangChain project
DEFAULT_COLLECTION = "docqa-agent"

print("REPO_ROOT:", REPO_ROOT)
print("DEFAULT_INDEX_DIR:", DEFAULT_INDEX_DIR)
print("DEFAULT_COLLECTION:", DEFAULT_COLLECTION)