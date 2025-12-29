DOCS ?= ./data
K ?= 5
MMR ?= 0
FETCH_K ?= 30
EMBEDDING ?= google
LLM_MODEL ?= google

COMMON := --docs $(DOCS) --k $(K) --embedding $(EMBEDDING) --llm-model $(LLM_MODEL)

MMR_FLAG :=
ifeq ($(MMR),1)
  MMR_FLAG := --mmr --fetch-k $(FETCH_K)
endif

.PHONY: help index index-rebuild retrieve run eval

help:
	@echo "Targets:"
	@echo "  make index            Load existing index"
	@echo "  make index-rebuild    Rebuild index from documents"
	@echo "  make retrieve         Debug retrieval output"
	@echo "  make run              Interactive QA loop"
	@echo "  make eval             Run evaluation suite"
	@echo ""
	@echo "Overrides:"
	@echo "  make run DOCS=./data K=10 MMR=1 FETCH_K=50 EMBEDDING=hf LLM_MODEL=google"

index:
	python -m main index $(COMMON)

index-rebuild:
	python -m main index $(COMMON) --rebuild-index

retrieve:
	python -m main retrieve $(COMMON) $(MMR_FLAG)

run:
	python -m main run $(COMMON) $(MMR_FLAG)

eval:
	python -m main eval --k $(K) --embedding $(EMBEDDING) --llm-model $(LLM_MODEL)