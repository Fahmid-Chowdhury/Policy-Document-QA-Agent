@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Windows task runner for Policy-Document-QA-Agent
REM ============================================================
REM Usage:
REM   make help
REM   make index
REM   make index-rebuild
REM   make retrieve
REM   make run
REM   make chat
REM   make eval
REM
REM Environment overrides:
REM   set DOCS=.\data
REM   set K=6
REM   set MMR=1
REM   set FETCH_K=30
REM   set EMBEDDING=google | hf
REM   set LLM_MODEL=google | hf
REM   set SESSION_ID=demo
REM   set NO_CITATIONS=1
REM   set DEBUG=1
REM ============================================================

if "%1"=="" goto help

REM Defaults
if "%DOCS%"=="" set DOCS=.\data
if "%K%"=="" set K=5
if "%MMR%"=="" set MMR=0
if "%FETCH_K%"=="" set FETCH_K=30
if "%EMBEDDING%"=="" set EMBEDDING=google
if "%LLM_MODEL%"=="" set LLM_MODEL=google
if "%SESSION_ID%"=="" set SESSION_ID=default
if "%NO_CITATIONS%"=="" set NO_CITATIONS=0
if "%DEBUG%"=="" set DEBUG=0

REM Common flags
set COMMON=--docs %DOCS% --k %K% --embedding %EMBEDDING% --llm-model %LLM_MODEL%

REM MMR flag
set MMR_FLAG=
if "%MMR%"=="1" set MMR_FLAG=--mmr --fetch-k %FETCH_K%

REM Optional flags
set DEBUG_FLAG=
if "%DEBUG%"=="1" set DEBUG_FLAG=--debug

set NO_CITATIONS_FLAG=
if "%NO_CITATIONS%"=="1" set NO_CITATIONS_FLAG=--no-citations

set SESSION_FLAG=--session-id %SESSION_ID%

if /I "%1"=="help" goto help
if /I "%1"=="index" goto index
if /I "%1"=="index-rebuild" goto index_rebuild
if /I "%1"=="retrieve" goto retrieve
if /I "%1"=="run" goto run
if /I "%1"=="chat" goto chat
if /I "%1"=="eval" goto eval

echo Unknown target: %1
goto help

:index
echo Running: python -m main index %COMMON% %DEBUG_FLAG%
python -m main index %COMMON% %DEBUG_FLAG%
goto end

:index_rebuild
echo Running: python -m main index %COMMON% --rebuild-index %DEBUG_FLAG%
python -m main index %COMMON% --rebuild-index %DEBUG_FLAG%
goto end

:retrieve
echo Running: python -m main retrieve %COMMON% %MMR_FLAG% %DEBUG_FLAG%
python -m main retrieve %COMMON% %MMR_FLAG% %DEBUG_FLAG%
goto end

:run
echo Running: python -m main run %COMMON% %MMR_FLAG% %NO_CITATIONS_FLAG% %DEBUG_FLAG%
python -m main run %COMMON% %MMR_FLAG% %NO_CITATIONS_FLAG% %DEBUG_FLAG%
goto end

:chat
echo Running: python -m main chat %COMMON% %MMR_FLAG% %SESSION_FLAG% %NO_CITATIONS_FLAG% %DEBUG_FLAG%
python -m main chat %COMMON% %MMR_FLAG% %SESSION_FLAG% %NO_CITATIONS_FLAG% %DEBUG_FLAG%
goto end

:eval
echo Running: python -m main eval --k %K% --embedding %EMBEDDING% --llm-model %LLM_MODEL% %DEBUG_FLAG%
python -m main eval --k %K% --embedding %EMBEDDING% --llm-model %LLM_MODEL% %DEBUG_FLAG%
goto end

:help
echo.
echo Targets:
echo   make index            Load existing index
echo   make index-rebuild    Rebuild index from documents
echo   make retrieve         Debug retrieval output
echo   make run              Interactive QA loop (single-turn)
echo   make chat             Conversational QA loop (multi-turn)
echo   make eval             Run evaluation suite
echo.
echo Environment overrides:
echo   set DOCS=.\data
echo   set K=6
echo   set MMR=1
echo   set FETCH_K=30
echo   set EMBEDDING=google ^| hf
echo   set LLM_MODEL=google ^| hf
echo   set SESSION_ID=demo
echo   set NO_CITATIONS=1
echo   set DEBUG=1
echo.
echo Examples:
echo   set EMBEDDING=hf ^&^& set LLM_MODEL=google ^&^& set SESSION_ID=demo ^&^& make chat
echo   set K=10 ^&^& set MMR=1 ^&^& make retrieve
echo   set DEBUG=1 ^&^& make chat
echo.
goto end

:end
endlocal
