from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status

from .serializers import IndexRequest, AskRequest, AskJsonRequest
from .services.docqa_service import DocQAConfig, rebuild_index, ask, ask_json
from .utils import ok, err
from .auth import HasAPIKey
from .services.docqa_service import _get_embeddings, _get_llm, _get_vectordb
from .safe import safe_api

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]  # Policy-Document-QA-Agent/
DEFAULT_INDEX_DIR = str(REPO_ROOT / "src" / ".index")  # keep index with your LangChain project
DEFAULT_COLLECTION = "docqa-agent"

CFG = DocQAConfig(
    index_dir=DEFAULT_INDEX_DIR,
    collection_name=DEFAULT_COLLECTION,
)

# Set your defaults here (match your CLI config values)
# CFG = DocQAConfig(
#     index_dir="./.index",           # relative to server working dir; you can make this absolute if you want
#     collection_name="docqa-agent",
# )


@api_view(["GET"])
@permission_classes([HasAPIKey])
@safe_api
def health(request):
    return Response({"status": "ok", "service": "docqa-api"})


@api_view(["POST"])
@permission_classes([HasAPIKey])
@safe_api
def index_endpoint(request):
    s = IndexRequest(data=request.data)
    if not s.is_valid():
        return err("Validation error", details=s.errors, http_status=status.HTTP_400_BAD_REQUEST)

    data = s.validated_data
    if not data.get("rebuild"):
        return ok({"message": "rebuild=false (no action)"})
    
    result = rebuild_index(CFG, docs_path=data["docs_path"], embedding=data["embedding"])
    return ok(result)


@api_view(["POST"])
@permission_classes([HasAPIKey])
@safe_api
def ask_endpoint(request):
    s = AskRequest(data=request.data)
    if not s.is_valid():
        return err("Validation error", details=s.errors, http_status=status.HTTP_400_BAD_REQUEST)
    
    data = s.validated_data
    result = ask(
        CFG,
        question=data["question"],
        k=data["k"],
        embedding=data["embedding"],
        llm_model=data["llm_model"],
    )
    return ok(result)


@api_view(["POST"])
@permission_classes([HasAPIKey])
@safe_api
def ask_json_endpoint(request):
    s = AskJsonRequest(data=request.data)
    if not s.is_valid():
        return err("Validation error", details=s.errors, http_status=status.HTTP_400_BAD_REQUEST)

    data = s.validated_data
    result = ask_json(
        CFG,
        question=data["question"],
        k=data["k"],
        embedding=data["embedding"],
        llm_model=data["llm_model"],
    )
    return ok(result)

@api_view(["POST"])
@permission_classes([HasAPIKey])
@safe_api
def warmup_endpoint(request):
    """
    Preload embeddings, LLM, and vectorstore into memory.
    """
    embedding = request.data.get("embedding", "google")
    llm_model = request.data.get("llm_model", "google")

    embeddings = _get_embeddings(embedding)
    _get_llm(llm_model)
    _get_vectordb(CFG, embeddings)

    return ok({"message": "warmed up", "embedding": embedding, "llm_model": llm_model})
