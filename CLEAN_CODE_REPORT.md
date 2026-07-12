# art-rium — Clean-Code- & Architektur-Bericht

*Stand: 2026-07-12 · Commit `2dacd36` · Fokus: DRY, SSOT, KISS, Performance, Feature-Velocity, Design-Anpassbarkeit*

Dieser Bericht baut auf dem Review vom 2026-06-09 ([CODE_REVIEW.md](CODE_REVIEW.md)) auf.
Er bewertet den **aktuellen** Stand (~24.500 Zeilen eigener Code) und ist auf drei Ziele
ausgerichtet:

1. **Performance** — spürbar schnelle UI und effiziente Backend-Pfade
2. **Feature-Velocity** — ein neues Tool / Feature in Stunden statt Tagen
3. **Design-Anpassbarkeit** — ein Redesign als 1-Datei-Änderung, nicht als 9-Datei-Marathon

---

## 1. Executive Summary

**Der Backend-Kern ist in gutem Zustand.** Seit dem Juni-Review wurde viel geliefert:
Hot-Path-Indexe + CHECK-Constraints per Migration, Companion-Posts als Kindtabelle,
die beiden God-Module (`ollama/client.py`, `wordpress/articles.py`) sauber gesplittet,
`safe_create_task` + Startup-Sweep gegen verlorene Jobs, 8 Testdateien mit CI,
Lockfile. Die Service-Schicht (`services/comfy|instagram|ollama|wordpress|youtube|video`)
ist klar geschnitten — neue Backend-Features finden dort schnell ihren Platz.

**Die drei größten offenen Hebel:**

| # | Befund | Trifft welches Ziel |
|---|---|---|
| 1 | **~3.000 Zeilen Inline-CSS über 9 Seiten dupliziert** — `header`, `.app`, Picker-Modal (3×), Progress-Bar (4×), Cards, Badges leben pro Tool statt in `shared.css` | Design-Anpassung, DRY/SSOT |
| 2 | **`routers/video.py` ist das neue God-Modul (1.608 Zeilen)** — 3 Workflow-Builder, 2 In-Memory-Job-Registries, Clips, Merge, Story-Mode, Soundtrack in einer Datei | Feature-Velocity, KISS |
| 3 | **Listen-Endpoints ohne Pagination + 29× Wegwerf-HTTP-Clients** | Performance |

Nichts davon ist ein Notfall. Alles davon wird mit jedem neuen Tool teurer — und die
Frontend-Duplikation ist der Grund, warum sich ein Redesign heute „riesig" anfühlt.

---

## 2. Was bereits richtig gut ist (bitte beim Refactoring erhalten)

- **Design-Tokens existieren und sind SSOT für Farben/Radien/Schatten**
  ([shared.css:11-60](frontends/shared/shared.css#L11-L60)): `--accent`, `--surface*`,
  `--radius*` etc. Ein Farbwechsel der Marke ist heute schon eine 1-Datei-Änderung.
  Das Problem liegt eine Ebene höher (Abschnitt 3).
- **`ArtRium`-Shared-Modul wird konsequent genutzt** ([shared.js](frontends/shared/shared.js)):
  alle 8 Tools binden `makeApiFetch` an, Toast/Dot/WebSocket sind zentral, 401-Recovery
  einheitlich. Kein Tool wickelt Auth selbst ab.
- **Service-Schicht sauber geschnitten**: `services/comfy/client.py` kapselt
  upload/post/poll/free; `services/ollama/*` trennt Transport, Analyse, Artikel,
  Validatoren; `services/wordpress/*` trennt Orchestrierung, Renderer, Gutenberg, SEO.
- **Datenmodell diszipliniert**: Composite-Index auf `(status, scheduled_at)`
  ([models.py:150](core/models.py#L150)), CHECK-Constraints auf Status-Spalten,
  `lazy="selectin"` verhindert N+1 ([models.py:195-205](core/models.py#L195-L205)),
  `list_posts` batch-lädt referenzierte Bilder/Videos statt pro Post zu queryen
  ([instagram.py:263-289](routers/instagram.py#L263-L289)).
- **`routers/images.py` zeigt das richtige Listen-Muster**: `limit`/`offset` mit
  Obergrenze ([images.py:34-44](routers/images.py#L34-L44)) — genau das fehlt den
  anderen Listen (Abschnitt 5.1).
- **Galerie nutzt Thumbnails + `loading="lazy"`** — Vollbild nur im Modal.
- **Tests & CI vorhanden**: 8 Testdateien für die riskanten reinen Funktionen
  (LLM-Salvage, ffmpeg-Builder, Gutenberg, Auth, Workflow-Builder) + GitHub-Workflow.

---

## 3. SSOT/DRY — Frontend (der Design-Hebel)

### 3.1 Diagnose: Die Tokens sind zentral, die Komponenten nicht

Jede der 9 Seiten trägt einen eigenen `<style>`-Block (129–757 Zeilen, zusammen
**~2.970 Zeilen Inline-CSS**). Eine Selektor-Häufigkeitsanalyse über alle Tools:

| Selektor | Definiert in | Sollte leben in |
|---|---|---|
| `header`, `header h1`, `.app` | **9 von 9 Seiten** | `shared.css` |
| `#toast` (Overrides) | 3 Seiten (trotz shared-Definition) | `shared.css` |
| `.picker-modal`, `.picker-grid`, `.picker-cell`, `.picker-header`, `.order-badge` | 3 Seiten (gallery, instagram, video) | Shared-Komponente |
| `.progress-track`/`.progress-fill`/`.progress-steps`/`.step-pill` | 3–4 Seiten | `shared.css` |
| `.card`, `.card h2`, `.section-label`, `.gen-btn`, `.status-badge`, `.sel-wrap` | je 2–3 Seiten | `shared.css` |

**Konsequenz heute:** „Header etwas höher, Cards etwas runder, Statusbadges neu" =
9 Dateien anfassen, 9× testen, und die Kopien driften (sie *sind* bereits gedriftet —
die drei Picker-Modals sind ähnlich, aber nicht identisch).

**Konsequenz nach dem Fix:** dieselbe Änderung = 1 Datei. Und ein neues Tool startet
mit ~150 statt ~400 Zeilen CSS.

### 3.2 Diagnose: JS-Scaffolding wird pro Tool neu geschrieben

- `escHtml` ist 2× lokal definiert ([instagram:1071](frontends/tools/instagram/index.html#L1071),
  [music:375](frontends/tools/music/index.html#L375)) — eine XSS-relevante Funktion
  gehört exakt einmal nach `shared.js`.
- **Bild-Picker-Logik 3× implementiert** (gallery, instagram, video): Grid laden,
  Auswahl-Reihenfolge, Order-Badges, Done-Button — jeweils leicht anders.
- **Job-Polling 3× handgerollt** (`setInterval` + fetch + Abbruchlogik in video 3×,
  improv, articles) — jedes Mal mit eigener Fehler-/Stop-Behandlung.
- Modal-Open/Close (Overlay, Escape, Scroll-Lock) pro Tool.

### 3.3 Empfehlung (bewusst ohne Build-Step, Philosophie bleibt)

**Stufe 1 — CSS-Komponentenschicht in `shared.css` (~1 Tag, größter Einzelhebel):**
`header`/`.app`-Layout, `.card`, `.section-label`, `.status-badge`, `.progress-*`,
`.step-pill`, `.picker-*` nach `shared.css` heben; die 9 Inline-Blöcke auf echte
Seiten-Spezifika eindampfen. Erwartung: Inline-CSS sinkt von ~2.970 auf unter
~1.200 Zeilen, und **ein Redesign wird zur 1-Datei-Änderung** (Tokens für Farben
gibt es ja schon — danach auch für Komponenten-Geometrie).

**Stufe 2 — Shared-JS-Komponenten (~2–3 Tage):**
`ArtRium` erweitern um:
- `escHtml`, `fmtDate` (Utilities)
- `openModal/closeModal` (ein Modal-Manager — steht schon als offener Punkt im Juni-Review)
- `createImagePicker({multi, onDone})` — ersetzt die 3 Kopien
- `pollJob({url, interval, onUpdate, isDone})` — ersetzt die 5+ Polling-Schleifen

**Stufe 3 — ES-Module (`app.js` pro Tool) statt Inline-`<script>`:**
`<script type="module" src="app.js">` — kein Bundler nötig, aber lintbar, testbar,
diffbar. Kann tool-weise passieren (beim nächsten Anfassen eines Tools migrieren,
kein Big Bang).

### 3.4 SSOT-Verstoß: Tool-Registry lebt an 3 Orten

Ein neues Tool erfordert heute: Verzeichnis anlegen + Eintrag in `_TOOL_NAMES`
([main.py:221](main.py#L221)) + handgeschriebene Tool-Card im Dashboard
([dashboard/index.html:239-337](frontends/dashboard/index.html#L239-L337)).

**Fix:** `main.py` scannt `frontends/tools/*` (Verzeichnis = gemountet), und jedes Tool
bekommt eine kleine `tool.json` (Name, Icon, Beschreibung, Reihenfolge), aus der das
Dashboard seine Cards rendert. Danach ist **„neues Tool" = 1 Verzeichnis**, sonst nichts.

---

## 4. DRY/KISS — Backend

### 4.1 `routers/video.py` ist das neue God-Modul — **HOCH (Velocity)**

1.608 Zeilen mit fünf Verantwortlichkeiten:

| Inhalt | Zeilen (ca.) | Gehört nach |
|---|---|---|
| 3 Workflow-Builder als Python-Node-Dicts (FLF2V, i2v, LTX) | [video.py:158-472](routers/video.py#L158-L472) | `services/comfy/video_workflows.py` |
| Generation-Runner (upload → submit → poll → persist, 3 Varianten) | [video.py:541-800](routers/video.py#L541-L800) | `services/video/generation.py` |
| Story-Frames-Pipeline + eigene Job-Registry | [video.py:998-1210](routers/video.py#L998-L1210) | `services/video/story.py` |
| Clips/Merge/Soundtrack | verteilt | bleiben, schrumpfen |
| HTTP-Endpoints | ~400 | Router (nur noch dünn) |

Der Router sollte nur Request-Validierung + Delegation enthalten. Das ist dieselbe
Operation wie der bereits gelungene `ollama/client.py`-Split — gleiche Schnitttechnik,
bekanntes Muster.

### 4.2 Drei parallele In-Memory-Job-Registries — **MITTEL (KISS)**

`_progress` in [video.py:89](routers/video.py#L89) und [music.py:48](routers/music.py#L48)
plus `_story_jobs` (Story-Mode) sind drei Kopien desselben Konzepts: dict, `_set_progress`,
manuelles `pop`, teils mit/teils ohne Pruning. Dazu kommen DB-Status-Spalten und der
WS-Listener als weitere Fortschrittskanäle.

**Fix jetzt:** ein `core/progress.py` (`set(key, phase, msg, pct)`, `get`, TTL-Eviction) —
ersetzt alle drei Registries, ~60 Zeilen.
**Fix später (R1c aus dem Juni-Review, weiterhin richtig):** kleine DB-Job-Tabelle als
einheitlicher Zustand für alle Hintergrundarbeit — dann gibt es genau *eine* Wahrheit
über „was läuft gerade", und das Dashboard kann sie anzeigen.

### 4.3 Submit-→-Poll-→-Persist-Choreographie 4× — **MITTEL**

`_post_workflow_with_retry` + `poll_history` + Progress-Updates + Persist ist in
`_run_flf2v_multi`, `_run_i2v_multi`, `_run_story_frames` und `routers/music.py`
nahezu identisch verdrahtet. Ein `run_comfy_job(client, workflow, on_progress)` in
`services/comfy/client.py` kollabiert das. **Jeder künftige ComfyUI-Workflow**
(neues Modell, neuer Modus) wird dann zu: Builder schreiben + einen Call — statt
80 Zeilen Choreographie zu kopieren.

### 4.4 Kleinere DRY-Punkte

- **`_serialize` 5× in Routern** (images, improv, instagram, music, video) — pro Modell
  ein `to_dict()` bzw. Pydantic-Response-Model; verhindert, dass Frontend-Felder je
  nach Endpoint driften.
- **Safe-Filename-`FileResponse`-Serving ~6×** ([generate.py](routers/generate.py),
  [music.py](routers/music.py), [video.py](routers/video.py)) — ein
  `serve_managed_file(dir, filename, media_type)` in `core/serving.py`.
- **Zwei ComfyUI-Clients**: [core/comfy.py](core/comfy.py) (`post_prompt`, wirft
  HTTPException) und [services/comfy/client.py](services/comfy/client.py)
  (`post_workflow`, wirft RuntimeError) tun dasselbe mit unterschiedlicher
  Fehlersemantik. Auf `services/comfy/client.py` konsolidieren; `core/comfy.py` wird
  ein dünner Wrapper oder verschwindet.
- **`workflows/`-Verzeichnis als SSOT unklar**: `video_ltx2_3_i2v (3).json`,
  drei `wan2_2`-Varianten — die Python-Builder sind die Wahrheit, die JSONs sind
  Referenz-Exporte. Umbenennen nach `workflows/reference/` + die Duplikate löschen,
  sonst editiert man in 6 Monaten die falsche Datei.

---

## 5. Performance

### 5.1 Listen ohne Pagination — **wächst zum echten Problem**

Diese Endpoints laden **alle** Zeilen und serialisieren sie pro Request:

- `GET /videos` — [video.py:1440-1444](routers/video.py#L1440-L1444)
- `GET /music` (Songs) — [music.py:392](routers/music.py#L392)
- `GET /video/clips` — [video.py:1318-1326](routers/video.py#L1318-L1326)
- `GET /instagram/posts` — [instagram.py:257](routers/instagram.py#L257) (immerhin batch-geladen)

Bei einer Generierungs-Pipeline, die täglich produziert, sind das in einem Jahr
tausende Zeilen pro Klick auf den Galerie-Tab. `images.py` hat das Muster schon
richtig (`limit=50, le=200` + offset) — **dasselbe Muster auf die vier Endpoints
kopieren**, Frontends laden nach (die Galerie kann bereits inkrementell rendern).

### 5.2 29× Wegwerf-`httpx.AsyncClient` — **günstiger Fix**

Jeder Ollama-/Graph-/WP-/Outpost-Call baut Client + Connection-Pool + TCP/TLS neu auf
(29 Fundstellen). Bei Poll-Schleifen (Graph-Container-Status, ComfyUI-History) ist das
ein neuer Handshake pro Tick.
**Fix:** `core/http.py` mit langlebigen Modul-Clients (`ollama_client`, `graph_client`,
`wp_client`), Timeouts pro Request statt pro Client. Zentralisiert nebenbei die
Timeout-Politik (SSOT) und macht Retry-Middleware später trivial. Lifespan-Shutdown
schließt sie.

### 5.3 Was Performance-seitig bereits stimmt

Indexe auf den Hot-Paths (Scheduler-Poll, Status-Filter), `selectin`-Loading statt N+1,
Thumbnails + Lazy-Loading in der Galerie, atomare ffmpeg-Writes, Streaming-Upload mit
Kappe, VRAM-Orchestrierung (Ollama-Unload + ComfyUI `/free`). Die teuren Operationen
(Generierung, Transcode) sind inhärent langsam und korrekt asynchron — dort ist nichts
zu holen.

### 5.4 Nicht tun

Kein Redis, kein Celery, kein Frontend-Framework, kein Bundler. Die Single-Prozess-
asyncio-Architektur trägt diese Last locker; jede dieser Abhängigkeiten würde
Feature-Velocity *senken*. Die Performance-Arbeit liegt in 5.1/5.2, nicht in
Infrastruktur.

---

## 6. Priorisierte Roadmap

**Jetzt — Stunden, direkter Ertrag**
1. Pagination auf `GET /videos`, `/music`, `/video/clips`, `/instagram/posts` (Muster aus `images.py`). *(Performance)*
2. `core/http.py` — geteilte httpx-Clients, 29 Fundstellen umstellen. *(Performance, SSOT für Timeouts)*
3. `escHtml`/`fmtDate` nach `shared.js`; `workflows/`-Duplikate aufräumen. *(DRY, Hygiene)*

**Als Nächstes — je ~1–3 Tage, die strategischen Hebel**
4. **CSS-Komponentenschicht in `shared.css`** (Header/App/Card/Badge/Progress/Picker) — danach ist Design-Anpassung eine 1-Datei-Änderung. *(SSOT/Design — größter Einzelhebel)*
5. `core/progress.py` — eine Job-Progress-Registry mit TTL statt drei Kopien. *(KISS)*
6. `video.py`-Split: Workflow-Builder → `services/comfy/video_workflows.py`, Runner → `services/video/`, Router wird dünn. *(Velocity)*
7. `run_comfy_job()`-Helper — Submit/Poll/Persist einmal. *(DRY, macht neue Workflows billig)*

**Später — architektonisch**
8. Shared-JS-Komponenten (Modal-Manager, Image-Picker, pollJob) + tool-weise Migration auf `app.js`-Module. *(Velocity/Design)*
9. Tool-Registry als SSOT: Verzeichnis-Scan + `tool.json` → Dashboard rendert Cards. Neues Tool = 1 Verzeichnis. *(Velocity)*
10. R1c aus dem Juni-Review: DB-Job-Tabelle für alle Hintergrundjobs → einheitliche Sicht „was läuft", Retry-Buttons, Dashboard-Integration. *(KISS, Observability)*
11. `_serialize`-Konsolidierung in Response-Models; `core/comfy.py` in `services/comfy/client.py` aufgehen lassen. *(DRY)*

**Messlatte für Erfolg** (in 3 Monaten prüfbar):
- Redesign-Probe: Akzentfarbe + Card-Radius + Header-Höhe ändern → **genau 1 Datei**.
- Neues-Tool-Probe: leeres Tool mit Liste + Detail + Job-Polling → **< 1 Tag, < 300 Zeilen**.
- Kein `index.html` über 1.000 Zeilen (heute: 4 Stück, Spitze 2.446).
- Kein Router über 600 Zeilen (heute: `video.py` 1.608).

---

## 7. Methodik

Vollständige Lektüre von `main.py`, `core/`, `shared.css`/`shared.js`, Struktur-Analyse
aller Router/Services/Worker, Selektor-Frequenzanalyse über alle 9 Frontend-Seiten,
Migrations- und Modell-Prüfung (Indexe, Constraints, Loading-Strategien), Stichproben
in den vier größten Frontends. Abgeglichen gegen CODE_REVIEW.md (2026-06-09), um
Erledigtes nicht erneut zu melden. Zeilenangaben beziehen sich auf Commit `2dacd36`.
