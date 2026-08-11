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

echo.
echo  =====================================
echo   art-rium  ^|  Remote Access
echo  =====================================
echo.

:: ── Start ComfyUI ────────────────────────────────────────────────────────────
:: --cuda-device 0 is REQUIRED on this box, not a tuning knob. With both GPUs
:: visible, comfy-aimdo (the dynamic weight streamer) inits for the 4060 Ti AND
:: the 2070 and then streams MiniMax H3's 15 GB text encoder to the wrong one —
:: the 8 GB 2070, which also drives the display. It dies ~3s into the load with
::   aimdo hostbuf_read_file_slice: device copy failed result=2 size=65536000
:: reported as "CUDA error: out of memory" while the 4060 Ti sits nearly empty,
:: and it takes ComfyUI's prompt_worker thread with it (server still answers,
:: /system_stats 500s, queue never advances).
:: Proven 2026-08-10: identical job, same image, same empty card — 4 crashes in
:: a row with both GPUs visible, then 6.0 min end-to-end with this flag.
:: Do not remove without re-testing a full MiniMax run.
::
:: NOTE: --fast-disk was tried here on 2026-08-09 and REMOVED the same day.
:: It routes dynamic weight loading through a disk-backed host buffer, and on
:: this box that path dies: a MiniMax H3 run failed 34s in with
::   aimdo hostbuf_read_file_slice: device copy failed result=2 size=67108864
:: taking ComfyUI's prompt worker with it (server still answers, queue never
:: advances). It also produced no measurable speed-up before it broke. Do not
:: re-add without re-testing a full MiniMax run.
set COMFY_DIR=E:\00_comfy
if exist "%COMFY_DIR%\venv\Scripts\activate.bat" (
    echo  Starting ComfyUI...
    start "ComfyUI" cmd /k cd /d "%COMFY_DIR%" ^&^& call venv\Scripts\activate.bat ^&^& python main.py --listen --cuda-device 0
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

echo  Cloudflare tunnel: https://art-rium.stefaneisele.com
echo  (cloudflared runs as a Windows service — always on)
echo.
pause

pause
