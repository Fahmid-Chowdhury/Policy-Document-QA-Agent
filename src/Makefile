DOCS ?= ./data
K ?= 5
MMR ?= 0
FETCH_K ?= 30
EMBEDDING ?= google
LLM_MODEL ?= google
SESSION_ID ?= default
NO_CITATIONS ?= 0
DEBUG ?= 0

COMMON := --docs $(DOCS) --k $(K) --embedding $(EMBEDDING) --llm-model $(LLM_MODEL)

MMR_FLAG :=
ifeq ($(MMR),1)
  MMR_FLAG := --mmr --fetch-k $(FETCH_K)
endif

NO_CITATIONS_FLAG :=
ifeq ($(NO_CITATIONS),1)
  NO_CITATIONS_FLAG := --no-citations
endif

DEBUG_FLAG :=
ifeq ($(DEBUG),1)
  DEBUG_FLAG := --debug
endif

SESSION_FLAG := --session-id $(SESSION_ID)

.PHONY: help index index-rebuild retrieve run chat eval

help:
	@echo "Targets:"
	@echo "  make index            Load existing index"
	@echo "  make index-rebuild    Rebuild index from documents"
	@echo "  make retrieve         Debug retrieval output"
	@echo "  make run              Interactive QA loop (single-turn)"
	@echo "  make chat             Conversational QA loop (multi-turn)"
	@echo "  make eval             Run evaluation suite"
	@echo ""
	@echo "Overrides:"
	@echo "  make chat DOCS=./data K=6 MMR=1 FETCH_K=30 EMBEDDING=hf LLM_MODEL=google SESSION_ID=demo DEBUG=1"

index:
	python -m main index $(COMMON) $(DEBUG_FLAG)

index-rebuild:
	python -m main index $(COMMON) --rebuild-index $(DEBUG_FLAG)

retrieve:
	python -m main retrieve $(COMMON) $(MMR_FLAG) $(DEBUG_FLAG)

run:
	python -m main run $(COMMON) $(MMR_FLAG) $(NO_CITATIONS_FLAG) $(DEBUG_FLAG)

chat:
	python -m main chat $(COMMON) $(MMR_FLAG) $(SESSION_FLAG) $(NO_CITATIONS_FLAG) $(DEBUG_FLAG)

eval:
	python -m main eval --k $(K) --embedding $(EMBEDDING) --llm-model $(LLM_MODEL) $(DEBUG_FLAG)
