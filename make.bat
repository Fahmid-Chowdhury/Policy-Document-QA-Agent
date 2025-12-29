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
REM   make eval
REM
REM Environment overrides:
REM   set DOCS=.\data
REM   set K=6
REM   set MMR=1
REM   set FETCH_K=30
REM   set EMBEDDING=google | hf
REM   set LLM_MODEL=google | hf
REM ============================================================

if "%1"=="" goto help

REM Defaults
if "%DOCS%"=="" set DOCS=.\data
if "%K%"=="" set K=5
if "%MMR%"=="" set MMR=0
if "%FETCH_K%"=="" set FETCH_K=30
if "%EMBEDDING%"=="" set EMBEDDING=google
if "%LLM_MODEL%"=="" set LLM_MODEL=google

REM Common flags
set COMMON=--docs %DOCS% --k %K% --embedding %EMBEDDING% --llm-model %LLM_MODEL%

REM MMR flag
set MMR_FLAG=
if "%MMR%"=="1" set MMR_FLAG=--mmr --fetch-k %FETCH_K%

if /I "%1"=="help" goto help
if /I "%1"=="index" goto index
if /I "%1"=="index-rebuild" goto index_rebuild
if /I "%1"=="retrieve" goto retrieve
if /I "%1"=="run" goto run
if /I "%1"=="eval" goto eval

echo Unknown target: %1
goto help

:index
echo Running: python -m main index %COMMON%
python -m main index %COMMON%
goto end

:index_rebuild
echo Running: python -m main index %COMMON% --rebuild-index
python -m main index %COMMON% --rebuild-index
goto end

:retrieve
echo Running: python -m main retrieve %COMMON% %MMR_FLAG%
python -m main retrieve %COMMON% %MMR_FLAG%
goto end

:run
echo Running: python -m main run %COMMON% %MMR_FLAG%
python -m main run %COMMON% %MMR_FLAG%
goto end

:eval
echo Running: python -m main eval --k %K% --embedding %EMBEDDING% --llm-model %LLM_MODEL%
python -m main eval --k %K% --embedding %EMBEDDING% --llm-model %LLM_MODEL%
goto end

:help
echo.
echo Targets:
echo   make index            Load existing index
echo   make index-rebuild    Rebuild index from documents
echo   make retrieve         Debug retrieval output
echo   make run              Interactive QA loop
echo   make eval             Run evaluation suite
echo.
echo Environment overrides:
echo   set DOCS=.\data
echo   set K=6
echo   set MMR=1
echo   set FETCH_K=30
echo   set EMBEDDING=google ^| hf
echo   set LLM_MODEL=google ^| hf
echo.
echo Examples:
echo   set EMBEDDING=hf ^&^& set LLM_MODEL=google ^&^& make run
echo   set K=10 ^&^& set MMR=1 ^&^& make retrieve
echo.
goto end

:end
endlocal