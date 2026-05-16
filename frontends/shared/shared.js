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

  /** ComfyUI node-ID → human-readable label (used by dashboard and z-image). */
  const NODE_LABELS = {
    '39': 'Loading CLIP…',
    '40': 'Loading VAE…',
    '41': 'Preparing latent…',
    '42': 'Zeroing conditions…',
    '44': 'Sampling…',
    '45': 'Encoding prompt…',
    '46': 'Loading model…',
    '47': 'Setting sampler…',
    '51': 'Loading LoRA…',
    '9':  'Decoding image…',
  };

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
   * @param {object}   opts
   * @param {string}   opts.clientId      - WS path segment
   * @param {function} opts.getApiKey     - called on every connect; returns current API key
   * @param {function} opts.onMessage     - called with the parsed JSON message object
   * @param {function} [opts.onConnecting]   - called just before each connect attempt
   * @param {function} [opts.onConnected]    - called on ws.onopen
   * @param {function} [opts.onDisconnected] - called on ws.onclose (before reconnect decision)
   * @param {function} [opts.on4001]         - called when server closes with code 4001 (auth failure)
   */
  const connectWs = ({
    clientId, getApiKey, onMessage,
    onConnecting, onConnected, onDisconnected, on4001,
  }) => {
    let ws;
    let stopped = false;

    const connect = () => {
      const apiKey = getApiKey();
      const proto  = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const key    = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : '';

      onConnecting?.();
      ws = new WebSocket(`${proto}//${location.host}/ws/${clientId}${key}`);

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

  // ── Public API ─────────────────────────────────────────────────────────────

  return {
    STORAGE_KEYS, NODE_LABELS,
    getClientId, getApiKey, saveApiKey, clearApiKey,
    getAuthHeaders, withAuth, apiFetch,
    toast, setDot, connectWs,
  };

})();
