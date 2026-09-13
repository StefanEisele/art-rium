/**
 * art-rium shared utilities — available on all pages as the global `ArtRium` object.
 *
 * Centralises: localStorage keys, auth helpers, toast, connection-dot, and the
 * managed WebSocket factory (connectWs) so every page's own script stays lean.
 */
const ArtRium = (() => {

  // ── Constants ──────────────────────────────────────────────────────────────

  /** localStorage keys shared by all tools (SSOT). */
  const STORAGE_KEYS = {
    clientId: 'z_client_id',
    apiKey:   'z_apikey',
  };

  // Stage labels for ComfyUI progress events used to live here as a node-ID
  // map, which only ever read correctly for the Z-Image workflow — ids are
  // workflow-local, so SDXL and Ernie showed "Node 12". The server now sends
  // the label with the event (services/comfy/node_labels.py).

  // ── Client-ID / API-key storage ────────────────────────────────────────────

  const getClientId = () => {
    let id = localStorage.getItem(STORAGE_KEYS.clientId);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(STORAGE_KEYS.clientId, id);
    }
    return id;
  };

  const getApiKey   = ()    => localStorage.getItem(STORAGE_KEYS.apiKey) || '';
  const saveApiKey  = (key) => localStorage.setItem(STORAGE_KEYS.apiKey, key);
  const clearApiKey = ()    => localStorage.removeItem(STORAGE_KEYS.apiKey);

  // ── Auth helpers ───────────────────────────────────────────────────────────

  const getAuthHeaders = (apiKey) => apiKey ? { 'X-API-Key': apiKey } : {};

  const withAuth = (url, apiKey) => {
    if (!apiKey) return url;
    return url + (url.includes('?') ? '&' : '?') + 'api_key=' + encodeURIComponent(apiKey);
  };

  /**
   * fetch() wrapper that injects auth headers and throws on non-2xx.
   *
   * @param {string}   url
   * @param {object}   [opts]     - standard fetch options
   * @param {string}   [apiKey]   - current API key
   * @param {function} [on401]    - optional callback for 401 responses
   */
  const apiFetch = async (url, opts = {}, apiKey = '', on401 = null) => {
    const r = await fetch(url, {
      ...opts,
      headers: { ...getAuthHeaders(apiKey), ...opts.headers },
    });
    if (r.status === 401 && on401) {
      on401();
      throw new Error('Authentication required');
    }
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(_fmtDetail(err.detail) || `Error ${r.status}`);
    }
    return r;
  };

  // FastAPI returns either a plain string in `detail` (HTTPException) or a
  // list of objects (RequestValidationError). The legacy `err.detail || ...`
  // would stringify a list to "[object Object]", which is the bug we'd see
  // on every 422 response. Normalise both shapes to a readable string.
  const _fmtDetail = (d) => {
    if (!d) return null;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d.map(e => {
        if (typeof e === 'string') return e;
        const loc = Array.isArray(e?.loc) ? e.loc.slice(1).join('.') : '';
        const msg = e?.msg || JSON.stringify(e);
        return loc ? `${loc}: ${msg}` : msg;
      }).join('; ');
    }
    return JSON.stringify(d);
  };

  /**
   * Bind apiFetch to a per-page API-key getter, with the standard 401 recovery
   * (clear the stale key, return to the dashboard's auth panel).
   * @param {function} getApiKey - returns the current key, e.g. () => state.apiKey
   */
  const makeApiFetch = (getApiKey) => (url, opts = {}) =>
    apiFetch(url, opts, getApiKey(), () => {
      clearApiKey();
      location.href = '/';
    });

  // ── HTML escaping ──────────────────────────────────────────────────────────

  /** Escape a value for safe interpolation into innerHTML (text and attributes). */
  const escHtml = (s) => {
    if (s == null) return '';
    return String(s)
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
  };

  // ── Toast notification ─────────────────────────────────────────────────────

  let _toastTimer;

  /**
   * Show the page's #toast element.
   * @param {string}  msg      - message text
   * @param {boolean} isError  - use error (red) styling when true
   */
  const toast = (msg, isError = false) => {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.className = 'show' + (isError ? ' error' : '');
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => { el.className = ''; }, 3500);
  };

  // ── Connection dot ─────────────────────────────────────────────────────────

  /**
   * Update the #dot element class to reflect WS status.
   * CSS classes `.dot.connecting / .connected / .disconnected` handle colours.
   * @param {'connecting'|'connected'|'disconnected'} status
   */
  const setDot = (status) => {
    const el = document.getElementById('dot');
    if (el) el.className = 'dot ' + status;
  };

  // ── Managed WebSocket ──────────────────────────────────────────────────────

  /**
   * Create a managed WebSocket connection with automatic reconnection.
   *
   * The returned object exposes:
   *   - `ready`      {boolean getter} — true when the socket is OPEN
   *   - `reconnect()` — close the current socket so it reconnects immediately
   *   - `stop()`      — close permanently (no auto-reconnect)
   *
   * Auth for the WS handshake rides the session cookie set at login (browsers
   * attach cookies to the handshake request same as any same-origin fetch) —
   * no API key needs to travel in the URL's query string.
   *
   * @param {object}   opts
   * @param {string}   opts.clientId      - WS path segment
   * @param {function} opts.onMessage     - called with the parsed JSON message object
   * @param {function} [opts.onConnecting]   - called just before each connect attempt
   * @param {function} [opts.onConnected]    - called on ws.onopen
   * @param {function} [opts.onDisconnected] - called on ws.onclose (before reconnect decision)
   * @param {function} [opts.on4001]         - called when server closes with code 4001 (auth failure)
   */
  const connectWs = ({
    clientId, onMessage,
    onConnecting, onConnected, onDisconnected, on4001,
  }) => {
    let ws;
    let stopped = false;

    const connect = () => {
      const proto  = location.protocol === 'https:' ? 'wss:' : 'ws:';

      onConnecting?.();
      ws = new WebSocket(`${proto}//${location.host}/ws/${clientId}`);

      ws.onopen  = () => { onConnected?.(); };
      ws.onerror = () => ws.close();
      ws.onclose = (e) => {
        onDisconnected?.();
        if (e.code === 4001) { on4001?.(); return; }
        if (!stopped) setTimeout(connect, 3500);
      };
      ws.onmessage = (e) => {
        try { onMessage(JSON.parse(e.data)); } catch (_) {}
      };
    };

    connect();

    return {
      get ready()  { return ws?.readyState === WebSocket.OPEN; },
      reconnect()  { stopped = false; if (ws) ws.close(); else connect(); },
      stop()       { stopped = true; ws?.close(); },
    };
  };

  // ── Session cache ──────────────────────────────────────────────────────────

  /**
   * Remember a tool's setup across page loads.
   *
   * Every tool here is a form you fill in before you press Generate: pictures
   * picked, a prompt per picture, a canvas, half a dozen dials. Leaving the
   * page threw all of it away, so stepping over to the Gallery to check one
   * thumbnail meant rebuilding the whole thing. This keeps it in localStorage
   * and puts it back on the way in.
   *
   * What is deliberately NOT cached is anything the server already owns — the
   * job list, the clip stacks, the picker's current page of images. Those are
   * fetched on load and would only go stale here.
   *
   * Two ways to say what to keep, because the tools are split on where they
   * hold it. `fields` is a list of element ids for values that live only in
   * the DOM (a textarea, a number input, a checkbox); `snapshot`/`restore`
   * carry structured state out of and back into the tool's `state` object.
   * Maps and Sets survive the round trip — several tools key their per-image
   * settings that way.
   *
   * Nothing is dispatched after a restore. Writing `.value` fires no event, and
   * that is the safe default: an `input` handler in these tools can kick off a
   * network call. Do any repainting in `restore`, where the tool can call its
   * own render function and knows what that costs.
   *
   * @param {string}   tool        - short id, unique per tool ("video", "z-image")
   * @param {object}   [opts]
   * @param {number}   [opts.version]  - bump to drop everything cached by an older shape
   * @param {string[]} [opts.fields]   - element ids whose value/checked is cached
   * @param {function} [opts.snapshot] - () => object, structured state to cache
   * @param {function} [opts.restore]  - (object) => void, put that state back
   * @param {number}   [opts.intervalMs] - autosave cadence while the page is visible
   */
  const session = (tool, {
    version = 1, fields = [], snapshot = null, restore = null, intervalMs = 1500,
  } = {}) => {
    const KEY = `artrium_session_${tool}`;
    // A cached payload is a convenience, never a reason to fill the disk. The
    // video tool's is a few KB; anything near this cap means something
    // server-owned crept into the snapshot.
    const MAX_BYTES = 512 * 1024;
    let lastWritten = null;
    let timer = null;
    // Once reset() has fired, nothing may write again. The page is on its way
    // to a reload, and a reload fires `pagehide` — so without this the flush
    // registered by start() puts the snapshot back a millisecond after it was
    // cleared, and Reset looks like it does nothing at all.
    let stopped = false;

    const el = (id) => document.getElementById(id);
    const readField  = (e) => (e.type === 'checkbox' ? e.checked : e.value);
    const writeField = (e, v) => { if (e.type === 'checkbox') e.checked = !!v; else e.value = v; };

    // Map and Set are the two structures these tools actually use and the two
    // JSON drops silently — a Map stringifies to "{}", which restores as an
    // empty object and takes every per-image prompt with it.
    const replacer = (_k, v) =>
      v instanceof Map ? {__t: 'Map', v: [...v]} :
      v instanceof Set ? {__t: 'Set', v: [...v]} : v;
    const reviver = (_k, v) =>
      v && v.__t === 'Map' ? new Map(v.v) :
      v && v.__t === 'Set' ? new Set(v.v) : v;

    const collect = () => ({
      v: version,
      fields: Object.fromEntries(
        fields.map((id) => [id, el(id) ? readField(el(id)) : undefined])
              .filter(([, v]) => v !== undefined)),
      state: snapshot ? snapshot() : undefined,
    });

    const flush = () => {
      if (stopped) return;
      let text;
      try { text = JSON.stringify(collect(), replacer); } catch (_) { return; }
      if (text === lastWritten) return;
      if (text.length > MAX_BYTES) {
        console.warn(`[session:${tool}] snapshot too large (${text.length} bytes) — not cached`);
        return;
      }
      try { localStorage.setItem(KEY, text); lastWritten = text; } catch (_) {}
    };

    return {
      /**
       * Put the cached setup back. Returns the payload, or null if there was
       * nothing usable.
       *
       * The payload comes back rather than a bare boolean because a `<select>`
       * whose options arrive from the server cannot be restored here — at load
       * time it has no matching option and assigning `.value` silently does
       * nothing. Those tools re-apply from the returned data once their list
       * has landed.
       */
      load() {
        let data;
        try {
          const raw = localStorage.getItem(KEY);
          if (!raw) return null;
          data = JSON.parse(raw, reviver);
        } catch (_) { return null; }
        // A shape change invalidates everything: half-restoring an old
        // snapshot is worse than starting clean, because the parts that did
        // restore look deliberate.
        if (!data || data.v !== version) { this.clear(); return null; }
        for (const [id, v] of Object.entries(data.fields || {})) {
          const e = el(id);
          if (e) writeField(e, v);
        }
        if (restore && data.state !== undefined) restore(data.state);
        return data;
      },

      /**
       * Begin autosaving. Polls rather than asking every mutation site to
       * report in — there are dozens of those per tool and one forgotten call
       * is a setting that silently does not stick. The write itself is skipped
       * unless the serialised snapshot actually changed.
       */
      start() {
        if (timer) return;
        timer = setInterval(() => { if (!document.hidden) flush(); }, intervalMs);
        // Leaving the page is exactly the moment this exists for, and it is
        // the moment the interval cannot cover. `pagehide` fires where
        // `beforeunload` does not (iOS Safari, and a PWA sent to the
        // background), and `visibilitychange` catches the app switch that
        // never becomes an unload at all.
        addEventListener('pagehide', flush);
        addEventListener('visibilitychange', () => { if (document.hidden) flush(); });
      },

      /** Write now rather than at the next tick. */
      save: flush,

      /** Forget the cache without touching the page. */
      clear() {
        try { localStorage.removeItem(KEY); } catch (_) {}
        lastWritten = null;
      },

      /**
       * Forget it and start over. Reloading is the honest way to get back to
       * defaults: the alternative is a per-tool "reset every field" routine
       * that drifts out of step with the form the moment a control is added.
       */
      reset() {
        stopped = true;
        clearInterval(timer);
        timer = null;
        this.clear();
        location.reload();
      },
    };
  };

  // ── Public API ─────────────────────────────────────────────────────────────

  return {
    STORAGE_KEYS,
    getClientId, getApiKey, saveApiKey, clearApiKey,
    getAuthHeaders, withAuth, apiFetch, makeApiFetch,
    escHtml, toast, setDot, connectWs, session,
  };

})();
