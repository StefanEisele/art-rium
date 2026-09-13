@echo off
:: ── Launch ComfyUI ───────────────────────────────────────────────────────────
:: The single source of truth for how ComfyUI is started on this box. Called by
:: start-remote.bat at boot, and by the dashboard's "ComfyUI neu starten" button
:: (services/comfy/control.py::restart). Both must produce a process with the
:: same flags, which is why the command lives here and not in either caller.
::
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
    exit /b 0
) else (
    echo  WARNING: ComfyUI not found at %COMFY_DIR% — skipping.
    exit /b 1
)
