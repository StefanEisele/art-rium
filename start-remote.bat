@echo off
setlocal enabledelayedexpansion
title art-rium Remote
cd /d "%~dp0"

:: ── Activate venv ─────────────────────────────────────────────────────────────
if not exist venv\Scripts\activate (
    echo  Run setup first: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause & exit /b 1
)
call venv\Scripts\activate

:: ── Install / sync dependencies ───────────────────────────────────────────────
echo  Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 ( echo  ERROR: pip install failed. & pause & exit /b 1 )

:: ── Load PORT from .env ───────────────────────────────────────────────────────
set PORT=8000
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if "%%a"=="PORT" set PORT=%%b
)

:: ── Check API key ─────────────────────────────────────────────────────────────
set HAS_KEY=0
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if "%%a"=="API_KEY" (
        if not "%%b"=="" (
            if not "%%b"=="your-generated-key-here" set HAS_KEY=1
        )
    )
)

if "%HAS_KEY%"=="0" (
    echo.
    echo  WARNING: No API_KEY set in .env!
    echo  Anyone who finds your tunnel URL can use your GPU.
    echo.
    set /p GENKEY="  Enter API key (or press Enter to generate one): "
    if "!GENKEY!"=="" (
        for /f "delims=" %%k in ('python -c "import secrets; print(secrets.token_hex(16))"') do set GENKEY=%%k
        echo  Generated key: !GENKEY!
    )
    python -c "import re, pathlib; p = pathlib.Path('.env'); t = p.read_text(); t = re.sub(r'^API_KEY=.*$', 'API_KEY=!GENKEY!', t, flags=re.MULTILINE); p.write_text(t)"
    echo  API key saved to .env
    echo.
)

:: ── Find cloudflared exe (current folder or PATH) ────────────────────────────
set CF=

if exist "%~dp0cloudflared.exe"                    set CF=%~dp0cloudflared.exe
if not defined CF if exist "%~dp0cloudflared-windows-386.exe"    set CF=%~dp0cloudflared-windows-386.exe
if not defined CF if exist "%~dp0cloudflared-windows-amd64.exe"  set CF=%~dp0cloudflared-windows-amd64.exe
if not defined CF if exist "%~dp0cloudflared-windows-arm64.exe"  set CF=%~dp0cloudflared-windows-arm64.exe

:: Fall back to PATH
if not defined CF (
    for /f "delims=" %%p in ('where cloudflared 2^>nul') do if not defined CF set CF=%%p
)

if not defined CF (
    echo.
    echo  cloudflared not found.
    echo  Download it from: https://github.com/cloudflare/cloudflared/releases
    echo  Save any cloudflared .exe into this folder and re-run.
    echo.
    pause & exit /b 1
)

echo  Using: %CF%

echo.
echo  =====================================
echo   art-rium  ^|  Remote Access
echo  =====================================
echo.

:: ── Start ComfyUI ────────────────────────────────────────────────────────────
set COMFY_DIR=E:\00_comfy
if exist "%COMFY_DIR%\venv\Scripts\activate.bat" (
    echo  Starting ComfyUI...
    start "ComfyUI" cmd /k cd /d "%COMFY_DIR%" ^&^& call venv\Scripts\activate.bat ^&^& python main.py
    echo  ComfyUI starting in background...
) else (
    echo  WARNING: ComfyUI not found at %COMFY_DIR% — skipping.
)

:: ── Stop any existing art-rium server on this port ───────────────────────────
echo  Checking for existing server on port %PORT%...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING" 2^>nul') do (
    echo  Stopping existing process ^(PID %%p^)...
    taskkill /PID %%p /F >nul 2>&1
)
taskkill /IM python.exe /FI "WINDOWTITLE eq art-rium server" /F >nul 2>&1
timeout /t 3 /nobreak >nul

echo  Starting server in HTTP mode ^(for tunnel^)...
start "art-rium server" cmd /k cd /d "%~dp0" ^&^& call venv\Scripts\activate ^&^& python main.py --http
echo  Waiting for server to be ready...
set TRIES=0
:wait_loop
timeout /t 2 /nobreak >nul
set /a TRIES+=1
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:%PORT%/api/health 2>nul | findstr "200" >nul
if not errorlevel 1 (
    echo  Server is up!
    goto server_ready
)
if %TRIES% geq 15 (
    echo  WARNING: Server not responding after 30s — starting tunnel anyway.
    goto server_ready
)
echo  Still waiting ^(%TRIES%/15^)...
goto wait_loop
:server_ready

echo  Starting cloudflared tunnel...
echo  The public HTTPS URL will appear below once ready.
echo.
echo  NOTE: Free quick tunnels give a NEW URL each time you restart.
echo  For a permanent URL, set up a named tunnel at dash.cloudflare.com.
echo.
echo  Tunnel URL (copy this to your phone):
echo.
"%CF%" tunnel --url http://127.0.0.1:%PORT%

pause
