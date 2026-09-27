@echo off
REM One-command launcher for the n8n side of the live demo (Windows).
REM
REM Starts n8n on port 5678 against the market_intel API on port 8000.
REM The workflows read MARKET_INTEL_* and SLACK_WEBHOOK_URL from the n8n
REM process environment via $env.* - this script loads them from the repo
REM root .env so secrets never live in this file. Install n8n once with:
REM   md "%USERPROFILE%\.mi-n8n" && cd /d "%USERPROFILE%\.mi-n8n" && npm install n8n
REM (pin to Node 22; n8n's sqlite native build fails on newer Node without
REM Visual Studio Build Tools).
setlocal
if not defined MI_N8N_HOME set "MI_N8N_HOME=%USERPROFILE%\.mi-n8n"
set "N8N_USER_FOLDER=%MI_N8N_HOME%\data"
set "GENERIC_TIMEZONE=UTC"
set "N8N_DIAGNOSTICS_ENABLED=false"
REM The Config (Set) nodes read $env.MARKET_INTEL_* / $env.SLACK_WEBHOOK_URL;
REM n8n denies env access from nodes unless this is explicitly allowed.
set "N8N_BLOCK_ENV_ACCESS_IN_NODE=false"

REM Load KEY=VALUE lines from the repo .env (comments via # are skipped).
if exist "%~dp0..\.env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%~dp0..\.env") do set "%%a=%%b"
)

cd /d "%MI_N8N_HOME%"
"%MI_N8N_HOME%\node.exe" "%MI_N8N_HOME%\node_modules\n8n\bin\n8n" start >> "%MI_N8N_HOME%\n8n.log" 2>> "%MI_N8N_HOME%\n8n.err.log"
