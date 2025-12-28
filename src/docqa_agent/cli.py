import argparse
import logging

from docqa_agent.config import load_config
from docqa_agent.logging_setup import setup_logging

logger = logging.getLogger("docqa_agent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docqa-agent",
        description="Production-grade Document QA Agent (CLI)",
    )

    parser.add_argument(
        "--docs",
        type=str,
        default=None,
        help="Path to documents folder (used in later phases).",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild vector index from documents (used in later phases).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of chunks to retrieve."
    )
    
    parser.add_argument(
        "--mmr",
        action="store_true",
        help="Use MMR retrieval (diverse results). If not set, uses similarity.",
    )
    
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=None,
        help="MMR candidate pool size (only used with --mmr).",
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to test retrieval. If omitted, a default query is used.",
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="health",
        choices=["health", "config", "ingest", "chunk", "index", "retrieve"],
        help="Command to run: health | config | ingest | chunk | index | retrieve",
    )
    return parser


def run_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config()

    log_level = "DEBUG" if args.debug else config.log_level
    setup_logging(log_level)

    logger.info("Starting docqa-agent")
    logger.debug("Args: %s", vars(args))

    if args.command == "health":
        print("OK: docqa-agent is running")
        return

    if args.command == "config":
        # Print config in a stable, readable way
        print("AppConfig:")
        print(f"  app_env   = {config.app_env}")
        print(f"  log_level = {config.log_level}")
        print(f"  index_dir = {config.index_dir}")
        return
    
    if args.command == "ingest":
        if not args.docs:
            raise SystemExit("Error: --docs is required for ingest")

        from docqa_agent.ingest import load_documents_from_folder

        docs = load_documents_from_folder(args.docs)

        print(f"Loaded documents: {len(docs)}")
        if docs:
            first = docs[0]
            print("First document metadata:")
            for k in ["source_file", "file_type", "page"]:
                print(f"  {k}: {first.metadata.get(k)}")
        return
    
    if args.command == "chunk":
        if not args.docs:
            raise SystemExit("Error: --docs is required for chunk")

        from docqa_agent.ingest import load_documents_from_folder
        from docqa_agent.chunking import chunk_documents

        docs = load_documents_from_folder(args.docs)
        chunks = chunk_documents(docs)

        print(f"Loaded documents: {len(docs)}")
        print(f"Total chunks: {len(chunks)}")

        if chunks:
            c = chunks[0]
            preview = c.page_content[:200].replace("\n", " ")
            print("First chunk preview:")
            print(f"  text: {preview}...")
            print("First chunk metadata:")
            for k in ["source_file", "file_type", "page", "chunk_id", "chunk_index"]:
                print(f"  {k}: {c.metadata.get(k)}")
        return

    if args.command == "index":
        if not args.docs:
            raise SystemExit("Error: --docs is required for index")

        from docqa_agent.ingest import load_documents_from_folder
        from docqa_agent.chunking import chunk_documents
        from docqa_agent.vectorstore import (
            # build_embeddings,
            build_embeddings_hf,
            build_or_load_chroma,
            rebuild_index_fresh,
            similarity_search,
        )

        # embeddings = build_embeddings()
        embeddings = build_embeddings_hf()

        # IMPORTANT: only build DB AFTER rebuild decision
        if args.rebuild_index:
            docs = load_documents_from_folder(args.docs)
            chunks = chunk_documents(docs)

            vectordb = rebuild_index_fresh(
                persist_dir=config.index_dir,
                collection_name=config.collection_name,
                embeddings=embeddings,
                chunks=chunks,
            )

            print(f"Rebuilt index with chunks: {len(chunks)}")
        else:
            vectordb = build_or_load_chroma(
                persist_dir=config.index_dir,
                collection_name=config.collection_name,
                embeddings=embeddings,
            )
            print("Loaded existing index (no rebuild).")

        # Sanity search
        query = "What is this document about?"
        results = similarity_search(vectordb, query=query, k=3)

        print("Top 3 results:")
        for i, r in enumerate(results, start=1):
            meta = r.metadata or {}
            src = meta.get("source_file")
            page = meta.get("page")
            cid = meta.get("chunk_id")
            preview = r.page_content[:120].replace("\n", " ")
            print(
                f"{i}. source={src} page={page} "
                f"chunk_id={cid} preview={preview}..."
            )

        return

    if args.command == "retrieve":
        if not args.docs:
            raise SystemExit("Error: --docs is required for retrieve")

        # from docqa_agent.vectorstore import build_embeddings, build_or_load_chroma
        from docqa_agent.vectorstore import build_embeddings_hf, build_or_load_chroma
        from docqa_agent.retriever import build_retriever, retrieve_docs

        embeddings = build_embeddings_hf()
        vectordb = build_or_load_chroma(
            persist_dir=config.index_dir,
            collection_name=config.collection_name,
            embeddings=embeddings,
        )

        question = args.query or "Summarize the main topic discussed in the documents."
        retriever = build_retriever(
            vectordb=vectordb,
            k=args.k,
            use_mmr=args.mmr,
            fetch_k=args.fetch_k,
        )
        docs = retrieve_docs(retriever, question)

        print(f"Query: {question}")
        print(f"Retrieved chunks: {len(docs)}")
        print(f"Mode: {'MMR' if args.mmr else 'similarity'} | k={args.k} | fetch_k={args.fetch_k}")

        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            src = meta.get("source_file")
            page = meta.get("page")
            cid = meta.get("chunk_id")
            preview = d.page_content[:180].replace("\n", " ")
            print(f"{i}. source={src} page={page} chunk_id={cid} preview={preview}...")
        return


