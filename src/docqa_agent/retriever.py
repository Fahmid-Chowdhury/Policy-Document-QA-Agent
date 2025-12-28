"""test cli command: python -m main retrieve --docs ./data --k 5 --mmr --fetch-k 30 --query "What is the main argument of the racism paper?"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma


def build_retriever(
    vectordb: Chroma,
    k: int = 5,
    use_mmr: bool = True,
    fetch_k: Optional[int] = None,
):
    """
    use_mmr=True:
      - More diverse results (less repetition).
      - fetch_k controls candidate pool size before diversity selection.
    """
    if fetch_k is None:
        fetch_k = max(20, k * 4)

    if use_mmr:
        return vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": fetch_k},
        )

    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_docs(retriever, question: str) -> List[Document]:
    # New LangChain style: invoke()
    return retriever.invoke(question)
