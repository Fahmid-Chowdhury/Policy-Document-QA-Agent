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
        "command",
        nargs="?",
        default="health",
        choices=["health", "config", "ingest", "chunk", "index"],
        help="Command to run: health | config | ingest | chunk | index",
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
            build_embeddings,
            build_or_load_chroma,
            rebuild_index,
            similarity_search,
        )

        embeddings = build_embeddings()
        vectordb = build_or_load_chroma(
            persist_dir=config.index_dir,
            collection_name=config.collection_name,
            embeddings=embeddings,
        )

        if args.rebuild_index:
            docs = load_documents_from_folder(args.docs)
            chunks = chunk_documents(docs)
            rebuild_index(vectordb, chunks)
            print(f"Rebuilt index with chunks: {len(chunks)}")
        else:
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
            print(f"{i}. source={src} page={page} chunk_id={cid} preview={preview}...")
        return

