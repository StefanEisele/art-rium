# art-rium — Senior Code Review & Improvement Report

*Review date: 2026-06-09 · Scope: full repository at commit `a06cd96`*

---

## 1. Executive Summary

art-rium is a self-hosted content-automation suite (~22,000 lines): a FastAPI backend
orchestrating ComfyUI (image / video / music generation), Ollama (VLM analysis, article
writing), WordPress publishing, Instagram scheduling (local + Raspberry-Pi "outpost"),
and YouTube uploads, with eight vanilla-JS PWA frontends.

**Overall assessment:** the codebase is in notably good shape for a single-maintainer
project. Code is consistently commented with *why*-level documentation, async patterns
are mostly correct, idempotency was clearly thought about (per-image upload locks,
`wp_media_id` re-checks), and operational lessons are encoded in comments (Cloudflare
tunnel timeouts, Ollama VRAM behaviour, AVIF quality sweet spots).

The main risks are not in the small — they are structural:

| Theme | Risk |
|---|---|
| **Zero automated tests, no CI** | Every refactor is a leap of faith; regressions only surface in production |
| **Single-process, in-memory job execution** | Generation/publish jobs die silently on restart or unhandled exception |
| **Security posture tuned for "trusted single user"** | API key travels in query strings, localStorage, and as the cookie value itself |
| **Two god-modules + one god-table** | `ollama/client.py` (1,772 LOC), `wordpress/articles.py` (1,413 LOC), `instagram_posts` (30+ columns, 5 parallel status columns) |
| **8 copy-pasted single-file frontends (~9,500 LOC)** | Shared UI fixes require 8 edits |

None of these are emergencies for a personal tool on a LAN behind a Cloudflare tunnel.
All of them become real the moment a second user, a second machine, or a second
maintainer appears.

---

## 2. What the Implementation Gets Right

Worth stating explicitly, because these strengths should be preserved during refactors:

- **Idempotent WordPress media upload** (`services/wordpress/media.py:42-83`): per-image
  `asyncio.Lock` with a re-fetch *inside* the lock before checking `wp_media_id`. Correct
  double-checked locking for a single process.
- **Defensive LLM JSON handling** (`services/ollama/client.py:65-178`): fence-stripping,
  per-prompt salvage repair functions, bounded retries, and error bodies surfaced from
  Ollama 5xx responses instead of being swallowed by `raise_for_status()`.
- **Streamed uploads with a hard cap** (`routers/improv.py:74-90`): 1 MB chunks, 1 GB
  limit, cleanup of the partial file on every failure path.
- **Atomic ffmpeg outputs**: transcodes write to a `.part` temp file and `Path.replace()`
  into place (`services/instagram/ig_video.py:107-130`) — readers never see a torn file.
- **Replay buffer for offline clients** (`workers/comfy_listener.py:79-87, 198-201`),
  with re-delivery only when live delivery failed (avoids duplicates on reconnect).
- **Sensible auth UX**: query-param key is promoted to an `HttpOnly` cookie and the URL
  is immediately cleaned via 303 redirect (`main.py:137-155`).
- **Honest comments**: `.env.example` and `core/config.py` document real failure modes
  (OAuth consent-screen token expiry, Cloudflare ~100 s upstream timeout vs. 2.5 min
  model cold-load).

---

## 3. Findings — Security

### S1 · Glob injection + recursive-scan DoS in public share endpoint — **HIGH**
`routers/generate.py:153-169` — `/share/image/{filename}` strips directory components
with `Path(filename).name`, but glob metacharacters (`*`, `?`, `[…]`) survive and are
passed to `rglob()`. Consequences:

- `GET /share/image/*?token=…` returns *some* file from managed storage or the raw
  ComfyUI output dir — enabling enumeration of files that were never meant to be shared.
- Every miss triggers a full recursive scan of `comfyui_output_dir` (potentially tens of
  thousands of files) — a cheap DoS amplifier on a *public, single-static-token* endpoint.
- With `IMAGE_SHARE_TOKEN` unset (documented "dev mode", `generate.py:146-150`), the
  endpoint is fully open.

**Fix:** validate the filename against `^[A-Za-z0-9._-]+$`, then do a direct
`(dir / name).resolve()` + `is_relative_to(dir)` existence check per search root —
no globbing. Consider per-file HMAC-signed URLs with expiry instead of one global token.

### S2 · API key as the universal bearer in query strings, localStorage and cookie value — **HIGH**
- `core/auth.py:40-46` accepts `?api_key=` on every endpoint; `frontends/shared/shared.js:50-53`
  (`withAuth`) deliberately appends it to media URLs; WebSocket auth is query-param-only
  (`shared.js:156`, `core/auth.py:70`). The key therefore lands in server access logs,
  proxy logs, and browser history.
- The auth cookie's *value is the raw API key* (`main.py:149`), so any log/leak of the
  cookie is a full credential leak, and the key cannot be rotated without breaking all
  sessions.
- `shared.js:42-43` persists the key in `localStorage` — readable by any XSS.
- Comparison uses `==` (`core/auth.py:55`) instead of `secrets.compare_digest()` —
  a (low-practicality) timing side channel.

**Fix (incremental):** (1) on login, exchange the key for a random opaque session token
stored server-side or HMAC-signed — cookie no longer contains the key; (2) WebSockets:
authenticate via the cookie (sent automatically on the WS handshake) and drop the
query-param path; (3) keep `?api_key=` only for `<img>`/`<video>` `src` URLs if needed,
or replace with short-lived signed URLs; (4) use `secrets.compare_digest`.

### S3 · Local-network auth bypass trusts header absence — **MEDIUM**
`core/auth.py:32-37` — requests from RFC-1918 addresses bypass auth *unless* a known
proxy header is present. Correct for the documented Cloudflare-tunnel topology, but any
other reverse proxy (nginx without `X-Forwarded-For` configured, a future container
network) makes every remote request appear local → full auth bypass. The contract is
implicit and fragile. **Fix:** make the bypass opt-in (`ALLOW_LAN_BYPASS=true`) and log
a startup warning when active.

### S4 · Instagram/Outpost credentials in URLs and unvalidated targets — **MEDIUM**
- `routers/instagram.py:691` and the Graph helpers pass `access_token` as a query param —
  standard for the Graph API, but it leaks into any HTTP-level logging; prefer request
  bodies where the API allows.
- `services/instagram/outpost.py:49-53` builds requests from `outpost_base_url` /
  `outpost_shared_secret` with no shape validation (HTTPS scheme, no control characters).

### S5 · Verbose error passthrough — **LOW**
e.g. `routers/generate.py:136` returns `f"{type(exc).__name__}: {exc}"` to the client.
Fine for a single-user tool; should become a generic message + server-side log line if
the audience ever widens.

### S6 · Default DB credentials — **LOW**
`core/config.py:16` ships `art_rium:changeme`. Emit a startup warning when the default
is detected.

---

## 4. Findings — Reliability & Correctness

### R1 · Fire-and-forget jobs are lost on restart and can fail silently — **HIGH**
The job model is `asyncio.create_task(...)` with no persistence and (in several places)
no done-callback:

- `routers/improv.py:101`, `workers/instagram_scheduler.py:106`, video/music generation
  tasks, `main.py:53` (warm-up).
- A server restart mid-generation orphans the job: ComfyUI finishes, but ingestion never
  runs and the DB row stays `generating`/`processing` forever. `instagram_scheduler.py:99`
  sets `reel_status="processing"` *before* spawning the task — if the task dies, the row
  is stuck in a state the scheduler never retries.
- Python additionally garbage-collects tasks nobody holds a reference to; a
  `safe_create_task()` wrapper that retains the task and logs exceptions via
  `add_done_callback` is a 20-line fix.

**Fix (staged):** (1) `safe_create_task()` everywhere, now; (2) a startup sweep that
re-queues or fails-out rows stuck in non-terminal states older than a threshold;
(3) longer-term, a tiny DB-backed job table (id, kind, payload, state, attempts) —
no need for Celery/Redis at this scale.

### R2 · Silent exception swallowing in the ComfyUI event loop — **HIGH**
`workers/comfy_listener.py:105-109` — `except Exception: pass` around `_route()` means a
DB outage during ingestion silently drops every completed image with no log line at all.
At minimum log the exception; ideally distinguish parse errors (skip) from ingest errors
(retry/alert).

### R3 · Unbounded in-memory state in the listener — **MEDIUM**
`comfy_listener.py:39-45` — `_prompt_meta` entries for jobs that error out *before* the
events arrive, and `_pending` replay buffers for clients that never reconnect, grow
without TTL for the life of the process. Add timestamps + periodic sweep (>24 h → drop).

### R4 · Concurrent transcodes of the same source race on one temp file — **MEDIUM**
`services/instagram/ig_video.py:107` — two simultaneous publishes of the same video both
write `<out>.mp4.part`; two ffmpeg processes interleave writes into one file. The final
`replace()` is atomic, but the content may be corrupt. **Fix:** per-output
`asyncio.Lock` (same pattern as `wordpress/media.py`), or unique temp names +
first-wins rename.

### R5 · Module-level progress dicts mutated by concurrent tasks — **MEDIUM**
`routers/video.py:53`, `routers/music.py:47` — duplicated `_progress` registries, no
locking, no eviction for abandoned jobs. Single-threaded asyncio makes individual dict
ops safe, but read-modify-write sequences across awaits are not, and the pattern is
duplicated. Extract a shared `core/progress.py` tracker with TTL eviction.

### R6 · Outpost dispatch loads all media into RAM — **MEDIUM**
`services/instagram/outpost.py:174, 214-215` — feed/reel dispatch `f.read()`s every
video fully into memory before the multipart POST; a 4-video carousel + reel can be
several hundred MB resident. Use file handles / streaming multipart, and add size checks.

### R7 · Per-call `httpx.AsyncClient` construction — **LOW**
`services/ollama/client.py:116` and several other modules create a fresh client (and
connection pool) per request. Correct, but wasteful; a module-level shared client with
per-request timeouts is cheaper and centralises timeout policy. *(Note: an earlier-pass
claim that retry timeouts were broken here is wrong — `timeout` is correctly applied per
attempt.)*

### R8 · Polling loops without per-request timeouts — **LOW–MEDIUM**
`services/instagram/graph.py:138` (container readiness) and similar poll loops rely on
client default timeouts; one hung GET can eat a large share of the poll budget. Pass an
explicit short per-request timeout inside loops.

### R9 · File + DB writes are not transactional — **LOW (accepted trade-off)**
Pattern throughout (e.g. `comfy_listener.py:241-277`, video save paths): file lands on
disk, then the DB insert may fail → orphan file (logged, kept). Acceptable, but a
periodic "orphan sweep" script would keep storage honest.

---

## 5. Findings — Architecture & Data Model

### A1 · God-modules — **HIGH (maintainability)**
- `services/ollama/client.py` — 1,772 LOC mixing transport, salvage helpers, image
  analysis, four article writers, validators, and German-revision logic.
  Split: `chat.py` (transport + `_chat_json`), `analysis.py`, `articles/` (per mode),
  `validators.py`.
- `services/wordpress/articles.py` — 1,413 LOC mixing orchestration, three Gutenberg
  renderers, category/tag resolution, SEO patching.
  Split: `orchestrator.py`, `renderers.py`, `gutenberg.py` (block builders), `seo.py`.

The three mode renderers (essay/work/lab) and validators share most of their structure —
a `Mode` dataclass (headings, label maps, block sequence) would collapse them.

### A2 · `instagram_posts` is a state-machine sprawl — **HIGH (maintainability)**
`core/models.py:131-187` — one table carries feed posts, reels, story companions, reel
companions, *and* outpost mirroring: five parallel status columns (`status`,
`story_status`, `reel_status`, `outpost_status`, `outpost_reel_status`), three
creation-id columns, two delay columns, two scheduled-at columns. Every new publish
target multiplies the matrix. **Direction:** extract companions into a child table
(`post_companions`: kind, delay, scheduled_at, status, media_id) — one state machine,
one shape, N rows.

### A3 · No DB-level integrity for the things that matter — **HIGH**
- **No secondary indexes anywhere.** The scheduler polls
  `WHERE status='scheduled' AND scheduled_at <= now()` every 60 s
  (`workers/instagram_scheduler.py`) — add indexes on `instagram_posts (status,
  scheduled_at)`, `videos.status`, `images.batch_id`, `articles.translation_group_id`.
  Cheap migration, future-proofs the hot paths.
- **Status columns are free-text** `String(32)` with allowed values living in comments
  (`models.py:88, 149, 158-172, 242, 287, 320`). One typo ("procesing") makes a row
  invisible to every query. Use PostgreSQL ENUMs or `CheckConstraint`.
- **ARRAY-of-UUID pseudo-relations** (`Article.image_ids`, `Video.image_ids`,
  `InstagramPost.reel_video_ids`) bypass FK integrity — deleting an image leaves
  dangling UUIDs that every consumer must defensively skip.
- `instagram_post_media` lacks `UniqueConstraint(post_id, position)` and a CHECK that
  exactly one of `image_id`/`video_id` is set (`models.py:190-213`).

### A4 · 8 copy-pasted single-file frontends — **HIGH (velocity)**
~9,500 lines across `frontends/tools/*/index.html` (580–2,444 each). `shared.js`/`shared.css`
exist and are good, but each tool still re-implements modals, toasts layout, list
rendering, upload flows, and per-tool `apiFetch` re-wrapping; rough estimate 25–40 % of
each file is scaffolding. A shared-UI bug is an 8-file fix.

The no-build-step philosophy is legitimate and worth keeping. Within it:
- extract more into `shared.js` (modal manager, list-with-thumbs component, upload
  widget, progress bar) and shared CSS components;
- adopt `<script type="module">` + small per-tool `app.js` files instead of inline
  scripts (testable, lintable, still no bundler);
- or introduce *one* dependency (htmx or Lit) — but only if tool count keeps growing.

### A5 · Duplicated router plumbing — **MEDIUM**
`_serialize()` (video/music/improv), safe-filename `FileResponse` serving (4×),
progress registries (2×), ComfyUI submit-and-poll choreography (video + music) are
near-identical. Extract `core/serving.py`, `core/progress.py`, and a
`comfy_job_runner()` helper.

### A6 · Inconsistent error semantics — **LOW**
404 vs 400 vs 409 for "exists but wrong state" varies by router
(`improv.py:60` vs `video.py`). Standardise (409 for state conflicts).

---

## 6. Findings — Engineering Process

### P1 · Zero tests — **HIGH**
No test directory, no test framework in `requirements.txt`. The highest-value, lowest-
effort targets (pure functions, no infra needed):
- `_strip_json_fences` + every salvage/validator function in `services/ollama/client.py`
  — this is hand-rolled parsing of adversarial LLM output, the textbook unit-test case;
- Gutenberg block renderers in `services/wordpress/articles.py` (string-in/string-out);
- `core/auth.py` (`is_local_ip`, key extraction precedence);
- ffmpeg command builders (assert argv contents without running ffmpeg);
- the share-endpoint filename validation (S1) once fixed.

Add `pytest` + `pytest-asyncio` + `httpx.ASGITransport` for router smoke tests.

### P2 · No CI — **MEDIUM**
No `.github/workflows`. A 20-line workflow running `ruff` + `pytest` on push would
catch import errors and regressions before they reach the machine that runs the show.

### P3 · Unpinned dependencies — **MEDIUM**
`requirements.txt` is floor-only (`fastapi>=0.111`). A fresh install next year resolves
to different majors. Add a lock (`pip-tools` / `uv lock`) alongside the loose spec.

### P4 · Repo hygiene — **LOW**
- `workflows/audio_ace_step_1_5_split (1).json` — duplicate-download artifact name.
- `guides/compass_artifact_wf-*.md` — opaque generated names.
- `start-remote.bat` implies a second deployment mode that is otherwise undocumented;
  a short `README.md` (setup, run modes, architecture sketch) is the single biggest
  onboarding win — the repo currently has none.

---

## 7. Prioritised Roadmap

**Now (small, high leverage — hours each)**
1. S1: lock down `/share/*` filename handling (regex + direct path check).
2. R2: log (don't swallow) exceptions in `ComfyListener._route`.
3. R1a: `safe_create_task()` wrapper with exception logging; use everywhere.
4. A3a: migration adding the four hot-path indexes.
5. P3: pin dependencies with a lockfile.

**Next (days)**
6. R1b: startup sweep for rows stuck in non-terminal status; mark failed/requeue.
7. S2: cookie carries a derived session token, not the raw key; WS auth via cookie.
8. A3b: ENUM/CHECK constraints on status columns; constraints on `instagram_post_media`.
9. P1: pytest harness + unit tests for LLM salvage/validators and Gutenberg renderers.
10. P2: minimal CI (ruff + pytest).
11. R4/R5: per-output transcode lock; shared progress tracker with TTL.

**Later (architectural)**
12. A1: split the two god-modules along the seams listed above.
13. A2: extract companion posts into a child table; collapse the status-column matrix.
14. A4/A5: shared frontend components + ES modules; dedupe router plumbing.
15. R1c: small DB-backed job table for all background work (uniform retry/observability).
16. R6: streaming multipart for outpost dispatch.

---

## 8. Product / Feature Opportunities

Beyond hardening, directions the existing architecture makes cheap:

- **Unified job dashboard.** Once jobs live in a DB table (R1c), a single "everything
  in flight" view (generation, uploads, scheduled posts, failures with retry buttons)
  replaces per-tool status polling — the dashboard frontend already exists as a shell.
- **Webhook/notification sink.** Post to ntfy/Telegram on job failure or publish
  success — currently failures are only discoverable by reading logs.
- **Content calendar view.** `instagram_posts.scheduled_at` + articles + YouTube uploads
  already contain everything needed for a week/month calendar across platforms.
- **Analytics backflow.** The Graph API and WP REST both expose insights; persisting
  reach/likes per `instagram_media_id` and views per `wp_post_id` would close the loop
  from generation → publication → performance, and could eventually feed prompt/style
  selection.
- **Semantic search over the library.** Images already get VLM descriptions at upload
  time; embedding those (pgvector) gives "find me images like…" across thousands of
  generations for near-zero marginal cost.
- **Batch re-analysis pipeline.** When the VLM model improves, a queued re-run of
  alt-text/SEO copy over existing media; the idempotent upload path already supports
  metadata PATCHes.
- **Multi-account abstraction.** Instagram credentials are global settings today; a
  small `accounts` table would unlock posting the same content to multiple profiles —
  prerequisite if the tool is ever offered to a second artist.

---

## 9. Methodology & Confidence

Review performed by reading the full backend (`main.py`, `core/`, `routers/`,
`services/`, `workers/`), the shared frontend layer, the data model and migrations, and
sampling the eight tool frontends and the WP mu-plugin. Three parallel deep-dive passes
were cross-checked; findings that did not survive verification were corrected or dropped
— notably: the WordPress media-upload "duplicate upload race" (the lock pattern is
correct), an alleged Ollama retry-timeout bug (timeouts are applied correctly), and the
share-endpoint issue originally reported as path traversal (directory components *are*
stripped; the residual issue is glob injection + scan cost, S1). Line numbers reference
commit `a06cd96`.
