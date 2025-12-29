import argparse
import logging
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace

from docqa_agent.config import load_config
from docqa_agent.logging_setup import setup_logging

load_dotenv()

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
        "--question",
        type=str,
        default=None,
        help="Question to ask (Phase 5). If omitted, uses a default.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="If provided, save structured JSON response to this file.",
    )
    
    parser.add_argument(
        "--simulate-parse-fail",
        action="store_true",
        help="Sanity test: simulate a JSON parse failure and verify fallback still returns valid JSON.",
    )
    
    parser.add_argument(
        "--no-citations",
        action="store_true",
        help="Start interactive mode with citations hidden.",
    )
    
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gemini-2.5-flash",
        help="LLM model to use for answering questions (Phase 5).",
    )
    
    parser.add_argument(
        "--embedding",
        type=str,
        default="google",
        help="Embedding model to use for vector store.",
    ) 

    parser.add_argument(
        "command",
        nargs="?",
        default="health",
        choices=["health", "config", "ingest", "chunk", "index", "retrieve", "ask", "ask_json", "run", "eval"],
        help="Command to run: health | config | ingest | chunk | index | retrieve | ask | ask_json | run | eval",
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

    """
    python main.py health
    python main.py --debug health
    """
    if args.command == "health":
        print("OK: docqa-agent is running")
        return

    """
    python main.py config
    """
    if args.command == "config":
        # Print config in a stable, readable way
        print("AppConfig:")
        print(f"  app_env   = {config.app_env}")
        print(f"  log_level = {config.log_level}")
        print(f"  index_dir = {config.index_dir}")
        return
    
    """
    python -m main ingest --docs ./data
    """
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
    
    """
    python -m main chunk --docs ./data
    """
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

    """
    rebuild: python -m main index --docs ./data --rebuild-index
    reload (no rebuild): python -m main index --docs ./data
    """
    if args.command == "index":
        if not args.docs:
            raise SystemExit("Error: --docs is required for index")

        from docqa_agent.ingest import load_documents_from_folder
        from docqa_agent.chunking import chunk_documents
        from docqa_agent.vectorstore import (
            build_embeddings,
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

    """
    similarity retrieval baseline: python -m main retrieve --docs ./data --k 5 --embedding hf --query "What are the leave policies?"
    MMR diversity check: python -m main retrieve --docs ./data --k 5 --mmr --fetch-k 30 --embedding hf --query "What are the leave policies?"
    """
    if args.command == "retrieve":
        if not args.docs:
            raise SystemExit("Error: --docs is required for retrieve")

        # from docqa_agent.vectorstore import build_embeddings, build_or_load_chroma
        from docqa_agent.vectorstore import build_embeddings_hf, build_or_load_chroma
        from docqa_agent.retriever import build_retriever, retrieve_docs

        if args.embedding == "google":
            embeddings = build_embeddings()
        elif args.embedding == "hf":
            embeddings = build_embeddings_hf()
        # embeddings = build_embeddings_hf()
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
    
    """
    answerable question: python -m main ask --docs ./data --k 6 --mmr --embedding hf --llm_model google --question "What are the leave policies?"
    unanswerable question (should refuse): python -m main ask --docs ./data --k 6 --mmr --embedding hf --llm_model google --question "What is the capital of Japan?"
    """
    if args.command == "ask":
        if not args.docs:
            raise SystemExit("Error: --docs is required for ask")

        from docqa_agent.vectorstore import build_embeddings, build_embeddings_hf, build_or_load_chroma
        from docqa_agent.retriever import build_retriever, retrieve_docs
        from docqa_agent.rag import build_llm, build_llm_hf, answer_question, INSUFFICIENT_MSG

        if args.embedding == "google":
            embeddings = build_embeddings()
        elif args.embedding == "hf":
            embeddings = build_embeddings_hf()
        # embeddings = build_embeddings()
        # embeddings = build_embeddings_hf()
        vectordb = build_or_load_chroma(
            persist_dir=config.index_dir,
            collection_name=config.collection_name,
            embeddings=embeddings,
        )

        question = args.question or "Summarize the main topic discussed in the documents."

        retriever = build_retriever(vectordb=vectordb, k=args.k, use_mmr=args.mmr, fetch_k=args.fetch_k)
        retrieved = retrieve_docs(retriever, question)

        # llm = build_llm()
        llm = ChatHuggingFace(llm = build_llm_hf())
        
        result = answer_question(llm, retrieved, question)

        print(f"Question: {question}\n")
        print("Answer:")
        print(result.answer_text)
        print("\nCitations (retrieved chunks):")
        if not result.citations:
            print("  (none)")
        else:
            for c in result.citations:
                print(f"  {c['chunk_tag']}: {c['source_file']} page={c['page']} chunk_id={c['chunk_id']}")
        return

    """
    answerable → valid JSON with citations: python -m main ask_json --docs ./data --k 6 --embedding hf --llm_model google --question "What are the leave policies?"
    unanswerable → refusal JSON: python -m main ask_json --docs ./data --k 6 --embedding hf --llm_model google --question "What is the capital of Japan?"
    """
    if args.command == "ask_json":
        if not args.docs:
            raise SystemExit("Error: --docs is required for ask_json")

        import json

        from docqa_agent.vectorstore import (
            build_embeddings,
            build_embeddings_hf,
            build_or_load_chroma,
            similarity_search_with_scores,
        )
        from docqa_agent.retriever import build_retriever, retrieve_docs
        from docqa_agent.structured_rag import build_llm, build_llm_hf, build_structured_answer
        from docqa_agent.schema import QAResponse

        if args.embedding == "google":
            embeddings = build_embeddings()
        elif args.embedding == "hf":
            embeddings = build_embeddings_hf()
        # embeddings = build_embeddings()
        # embeddings = build_embeddings_hf()
        vectordb = build_or_load_chroma(
            persist_dir=config.index_dir,
            collection_name=config.collection_name,
            embeddings=embeddings,
        )

        question = args.question or "Summarize the main topic discussed in the documents."

        # Get docs + scores (confidence heuristic)
        scored = similarity_search_with_scores(vectordb, question, k=args.k)
        retrieved_docs = [d for (d, s) in scored]
        scores = [float(s) for (d, s) in scored]

        # Optional: compare retriever vs direct top-k scoring results
        # (Keeping it simple for now: we use scored top-k)
        
        if args.llm_model == "google":
            llm = build_llm()
        elif args.llm_model == "hf":
            llm = ChatHuggingFace(llm = build_llm_hf())
        # llm = build_llm()
        # llm = ChatHuggingFace(llm = build_llm_hf())
        

        if args.simulate_parse_fail:
            # Force fallback path while still returning valid JSON
            result = QAResponse(
                question=question,
                answer="Insufficient evidence in the provided documents.",
                citations=[],
                confidence=0.0,
                insufficient_evidence=True,
            )
        else:
            result = build_structured_answer(
                llm=llm,
                question=question,
                retrieved_docs=retrieved_docs,
                retrieved_scores=scores,
            )

        payload = result.model_dump()

        print(json.dumps(payload, indent=2, ensure_ascii=False))

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"\nSaved JSON to: {args.out}")

        return
    
    """
    with index rebuild: python -m main run --docs ./data --k 15 --mmr --rebuild-index --llm-model google --embedding hf
    without index rebuild: python -m main run --docs ./data --k 15 --mmr --llm-model google --embedding hf
    """
    if args.command == "run":
        if not args.docs:
            raise SystemExit("Error: --docs is required for run")

        from docqa_agent.ingest import load_documents_from_folder
        from docqa_agent.chunking import chunk_documents
        from docqa_agent.vectorstore import (
            build_embeddings,
            build_embeddings_hf,
            build_or_load_chroma,
            rebuild_index_fresh,
            similarity_search_with_scores,
        )
        from docqa_agent.structured_rag import build_llm, build_llm_hf, build_structured_answer
        from docqa_agent.interactive import SessionState, handle_command, print_help

        if args.embedding == "google":
            embeddings = build_embeddings()
        elif args.embedding == "hf":
            embeddings = build_embeddings_hf()
        # embeddings = build_embeddings()
        # embeddings = build_embeddings_hf()

        # IMPORTANT: avoid Windows lock issue by deciding rebuild BEFORE opening DB
        if args.rebuild_index:
            docs = load_documents_from_folder(args.docs)
            chunks = chunk_documents(docs)
            vectordb = rebuild_index_fresh(
                persist_dir=config.index_dir,
                collection_name=config.collection_name,
                embeddings=embeddings,
                chunks=chunks,
            )
            print(f"Index rebuilt with chunks: {len(chunks)}")
        else:
            vectordb = build_or_load_chroma(
                persist_dir=config.index_dir,
                collection_name=config.collection_name,
                embeddings=embeddings,
            )
            print("Index loaded.")

        if args.llm_model == "google":
            llm = build_llm()
        elif args.llm_model == "hf":
            llm = ChatHuggingFace(llm = build_llm_hf())
        # llm = build_llm()
        # llm = ChatHuggingFace(llm = build_llm_hf())
        

        state = SessionState(show_citations=(not args.no_citations), last_response=None)
        print_help()
        print("\nInteractive mode. Ask questions or type :help\n")

        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return

            if not line:
                continue

            if handle_command(state, line):
                continue

            question = line

            # Retrieve docs + scores for confidence
            scored = similarity_search_with_scores(vectordb, question, k=args.k)
            retrieved_docs = [d for (d, s) in scored]
            scores = [float(s) for (d, s) in scored]

            result = build_structured_answer(
                llm=llm,
                question=question,
                retrieved_docs=retrieved_docs,
                retrieved_scores=scores,
            )

            state.last_response = result

            print("\nAnswer:\n")
            print(result.answer)

            if state.show_citations:
                print("\nCitations:\n")
                if not result.citations:
                    print("  (none)")
                else:
                    for c in result.citations:
                        # This assumes your Phase 6 schema is chunk-level (source_file/page/chunk_id/quote)
                        print(f"  - {c.source_file} page={c.page} chunk_id={c.chunk_id}")
                        # print(f"    quote: {c.quote}")

            print("")  # spacing
    
    """
    python -m main eval --k 10 --embedding hf --llm-model google
    """
    if args.command == "eval":
        from docqa_agent.eval import main as eval_main

        # Use config index settings; k from CLI
        code = eval_main(index_dir=config.index_dir, collection_name=config.collection_name, k=args.k, embedding=args.embedding, llm_model=args.llm_model)
        raise SystemExit(code)
