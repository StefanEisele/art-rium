/* ─────────────────────────────────────────────────────────────────────────────
   Embedding trainer — shared by the Embeddings tool and the Struktur tool.

   kentskooking's look comes from embeddings he trained on ~20 of his own
   pictures; this is the form and the list that make one from a gallery series
   (or a hand-picked set) and let the best snapshot be chosen. It renders its
   own markup into a container, so the two pages carry no copy of it:

     const trainer = ArtRium.EmbeddingTrainer.mount(el, {
       api, withAuth,                     // the page's fetch + URL helpers
       onLoad(lib) {},                    // every refresh of GET /api/embeddings
       onUse(training) {}, useLabel,      // optional action on a finished one
       onLadder(training) {},             // optional snapshot ladder
       pickImages: true,                  // offer "Bilder wählen" beside series
       formOpen: false,                   // the form fold's initial state
     });
     trainer.reload(); trainer.stop();

   Numbers quoted in the UI come from the server (`options`), which has them
   from measurement: 3.5 steps/s on the 4060 Ti, lr 5e-3, a snapshot every 250.
   ───────────────────────────────────────────────────────────────────────────── */
(function () {
  const TEMPLATES = [
    { key: 'style', label: 'Stil', hint: 'Lernt den Look der Bilder — Palette, Licht, Inszenierung.' },
    { key: 'object', label: 'Objekt', hint: 'Lernt ein Ding, das in allen Bildern vorkommt.' },
  ];
  const STATUS = {
    queued: 'wartet', training: 'trainiert', done: 'fertig',
    failed: 'fehlgeschlagen', cancelled: 'abgebrochen',
  };
  const PICK_PAGE = 60;

  const esc = (s) => ArtRium.escHtml(s == null ? '' : String(s));
  const fmtMin = (secs) => secs >= 90 ? `${Math.round(secs / 60)} min` : `${Math.round(secs)} s`;
  const running = (t) => t.status === 'queued' || t.status === 'training';

  // Mirrors services/embedding/plan.py::normalize_name for the preview; the
  // server's answer is the one that counts.
  function slugify(text) {
    return (text || '').toLowerCase()
      .replace(/ä/g, 'ae').replace(/ö/g, 'oe').replace(/ü/g, 'ue').replace(/ß/g, 'ss')
      .replace(/[\s.]+/g, '-').replace(/[^a-z0-9_-]/g, '').replace(/^[-_]+|[-_]+$/g, '')
      .slice(0, 40);
  }

  function chip(text, onclick, kind) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip-btn' + (kind ? ' ' + kind : '');
    b.textContent = text;
    b.onclick = onclick;
    return b;
  }

  const FORM = `
    <details class="fold et-form">
      <summary><span class="fold-title">Neues Embedding trainieren</span>
        <span class="fold-value et-form-value"></span></summary>
      <div class="fold-body">
        <div class="et-field et-source-field">
          <span class="et-label">Quelle</span>
          <div class="et-tabs et-source"></div>
        </div>
        <div class="et-field et-series-field">
          <span class="et-label">Bildserie</span>
          <select class="et-series"></select>
          <div class="et-hint et-series-hint"></div>
        </div>
        <div class="et-field et-pick-field">
          <span class="et-label">Bilder <b class="et-pick-count"></b></span>
          <input type="search" class="et-pick-search" placeholder="Prompt durchsuchen…" autocomplete="off">
          <div class="et-pick-grid"></div>
          <button type="button" class="chip-btn et-pick-more">Mehr laden</button>
        </div>
        <div class="et-field">
          <span class="et-label">Name</span>
          <input type="text" class="et-name" placeholder="z. B. rostbluete" autocomplete="off">
          <div class="et-hint et-name-hint"></div>
        </div>
        <div class="et-field">
          <span class="et-label">Art</span>
          <div class="et-tabs et-template"></div>
          <div class="et-hint et-template-hint"></div>
        </div>
        <div class="et-field">
          <div class="et-row"><span class="et-label">Vektoren</span><b class="et-vectors-val">8</b></div>
          <input type="range" class="et-vectors" min="1" max="75" step="1" value="8">
          <div class="et-hint">Mehr Vektoren = mehr Kapazität, und mehr Gewicht im Prompt.
            <button type="button" class="et-link et-kent">kentskooking: 70</button></div>
        </div>
        <div class="et-field">
          <span class="et-label">Schritte</span>
          <select class="et-steps">
            <option value="1000">1000</option><option value="2000" selected>2000</option>
            <option value="3000">3000</option><option value="5000">5000</option>
          </select>
          <div class="et-hint et-eta"></div>
        </div>
        <details class="fold quiet">
          <summary><span class="fold-title">Erweitert</span></summary>
          <div class="fold-body">
            <div class="et-field">
              <span class="et-label">Lernrate</span>
              <input type="number" class="et-lr" step="0.0005" min="0.0001" max="0.02" value="0.005">
              <div class="et-hint">5e-3 (A1111-Standard). Der Wert aus der sd-scripts-Doku, 1e-6, lernt nichts.</div>
            </div>
            <div class="et-field">
              <span class="et-label">Startwort</span>
              <input type="text" class="et-init" placeholder="painting" autocomplete="off">
              <div class="et-hint">Ein englisches Wort, von dem die Vektoren starten.</div>
            </div>
          </div>
        </details>
        <button type="button" class="btn primary et-start">Training starten</button>
        <div class="et-hint">Belegt die 4060 Ti für die Dauer — währenddessen nichts rendern.</div>
      </div>
    </details>
    <div class="et-list-title">Trainings</div>
    <div class="et-list"></div>`;

  function mount(root, opts) {
    const o = Object.assign({
      onLoad: null, onUse: null, useLabel: 'Verwenden', onLadder: null,
      pickImages: false, formOpen: false,
    }, opts);
    const api = o.api;
    const withAuth = o.withAuth;
    const st = {
      lib: null, series: null, poll: null, template: 'style', source: 'series',
      picked: new Map(), pickPage: 0, pickDone: false, pickSearch: '', pickLoading: false,
    };
    root.classList.add('et');
    root.innerHTML = FORM;
    const q = (sel) => root.querySelector(sel);
    if (o.formOpen) q('.et-form').open = true;
    if (!o.pickImages) q('.et-source-field').style.display = 'none';

    // ── Library / list ─────────────────────────────────────────────────────
    async function reload() {
      try {
        st.lib = await (await api('/api/embeddings')).json();
      } catch (e) {
        q('.et-list').innerHTML = `<div class="et-hint">Konnte Trainings nicht laden: ${esc(e.message)}</div>`;
        return null;
      }
      renderList();
      renderForm();
      if (o.onLoad) o.onLoad(st.lib);
      clearInterval(st.poll);
      st.poll = st.lib.trainings.some(running) ? setInterval(poll, 3000) : null;
      return st.lib;
    }

    async function poll() {
      const before = (st.lib ? st.lib.trainings : []).filter(running).map(t => t.id);
      await reload();
      for (const id of before) {
        const t = st.lib && st.lib.trainings.find(x => x.id === id);
        if (t && t.status === 'done') ArtRium.toast(`Embedding „${t.name}“ ist fertig.`);
        if (t && t.status === 'failed') ArtRium.toast(`Training „${t.name}“ fehlgeschlagen.`, true);
      }
    }

    function stop() { clearInterval(st.poll); st.poll = null; }

    function renderList() {
      const list = q('.et-list');
      const rows = st.lib.trainings;
      list.innerHTML = rows.length ? '' : '<div class="et-hint">Noch kein Training.</div>';
      for (const t of rows) list.appendChild(card(t));
    }

    function card(t) {
      const el = document.createElement('div');
      el.className = 'et-card';
      el.innerHTML =
        `<div class="et-card-head"><b>${esc(t.name)}</b>` +
        `<span class="et-status s-${t.status}">${STATUS[t.status] || t.status}</span></div>` +
        `<div class="et-meta">${t.image_count} Bilder · ${t.vectors} Vektoren · ${t.steps} Schritte · ` +
        `${t.seconds ? fmtMin(t.seconds) : '≈ ' + fmtMin(t.estimate_seconds)}` +
        `${t.status === 'done' ? ' · <code>artrium/' + esc(t.name) + '</code>' : ''}</div>`;
      if (running(t)) {
        const live = t.live || {};
        const prog = document.createElement('div');
        prog.innerHTML =
          `<div class="et-row"><span>${esc(live.message || 'Läuft…')}</span><span>${live.pct || 0}%</span></div>` +
          `<div class="et-track"><div class="et-fill" style="width:${live.pct || 0}%"></div></div>`;
        el.appendChild(prog);
      }
      if (t.error && t.status !== 'done') {
        const err = document.createElement('div');
        err.className = 'et-error';
        err.textContent = t.error;
        el.appendChild(err);
      }
      if (t.snapshots && t.snapshots.length) el.appendChild(snapshots(t));

      const actions = document.createElement('div');
      actions.className = 'et-actions';
      if (t.status === 'done') {
        if (o.onUse) actions.appendChild(chip(o.useLabel, () => o.onUse(t), 'apply'));
        if (o.onLadder) actions.appendChild(chip('Snapshot-Leiter', () => o.onLadder(t)));
      }
      if (running(t)) actions.appendChild(chip('Abbrechen', () => cancel(t), 'remove'));
      else actions.appendChild(chip('Löschen', () => remove(t), 'remove'));
      el.appendChild(actions);
      return el;
    }

    // One row per snapshot, its previews side by side at fixed seeds, so a
    // column reads as "the same prompt, as the embedding learns". Step 0 is the
    // untrained init word — the "before" every other row is read against.
    function snapshots(t) {
      const wrap = document.createElement('div');
      wrap.className = 'et-snaps';
      const prompts = (st.lib.options && st.lib.options.sample_prompts) || [];
      const head = document.createElement('div');
      head.className = 'et-snap et-snap-head';
      head.innerHTML = '<span></span>' + [0, 1, 2].map(i =>
        `<span>${esc((prompts[i] || '').split(' ').slice(0, 3).join(' '))}…</span>`).join('') + '<span></span>';
      wrap.appendChild(head);
      const last = Math.max(...t.snapshots.map(s => s.step));
      for (const snap of t.snapshots) {
        const row = document.createElement('div');
        row.className = 'et-snap';
        const active = t.status === 'done' &&
          (t.chosen_step === snap.step || (t.chosen_step == null && snap.step === last));
        const lbl = document.createElement('div');
        lbl.className = 'et-snap-lbl';
        lbl.innerHTML = (active ? '<b>aktiv</b>' : '') + (snap.step === 0 ? 'vorher' : snap.step);
        row.appendChild(lbl);
        for (let i = 0; i < 3; i++) {
          const url = snap.samples[i];
          if (!url) { row.appendChild(document.createElement('span')); continue; }
          const img = document.createElement('img');
          img.src = withAuth(url);
          img.loading = 'lazy';
          img.alt = '';
          img.onclick = () => window.open(withAuth(url), '_blank');
          row.appendChild(img);
        }
        row.appendChild(t.status === 'done' && snap.embedding && !active
          ? chip('verwenden', () => choose(t, snap.step))
          : document.createElement('span'));
        wrap.appendChild(row);
      }
      return wrap;
    }

    async function choose(t, step) {
      try {
        await api(`/api/embeddings/trainings/${t.id}/choose`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ step }),
        });
        ArtRium.toast(`artrium/${t.name} nutzt jetzt Schritt ${step}.`);
        await reload();
      } catch (e) { ArtRium.toast(e.message || 'Konnte Snapshot nicht setzen', true); }
    }

    async function cancel(t) {
      if (!confirm(`Training „${t.name}“ abbrechen?`)) return;
      try {
        await api(`/api/embeddings/trainings/${t.id}/cancel`, { method: 'POST' });
        await reload();
      } catch (e) { ArtRium.toast(e.message || 'Abbrechen fehlgeschlagen', true); }
    }

    async function remove(t) {
      if (!confirm(`„${t.name}“ löschen?\n\nEmbedding, alle Snapshots und Vorschauen werden entfernt.`)) return;
      try {
        await api(`/api/embeddings/trainings/${t.id}`, { method: 'DELETE' });
        await reload();
      } catch (e) { ArtRium.toast(e.message || 'Löschen fehlgeschlagen', true); }
    }

    // ── Form ───────────────────────────────────────────────────────────────
    async function loadSeries() {
      try {
        st.series = await (await api('/api/series?limit=200')).json();
      } catch { st.series = []; }
      const sel = q('.et-series');
      sel.innerHTML = st.series.length ? ''
        : '<option value="">Keine Serien — in der Galerie anlegen oder Bilder wählen</option>';
      for (const s of st.series) {
        const opt = document.createElement('option');
        opt.value = s.id;
        opt.textContent = `${s.title || 'Ohne Titel'} · ${s.item_count} Bilder`;
        sel.appendChild(opt);
      }
      suggestName(true);
    }

    const currentSeries = () => (st.series || []).find(x => x.id === q('.et-series').value);
    const imageCount = () => st.source === 'series'
      ? ((currentSeries() || {}).item_count || 0) : st.picked.size;

    function suggestName(force) {
      // The obvious name is the series title; offered only while the field is
      // still the previous suggestion, so a typed name is never overwritten.
      const name = q('.et-name');
      const s = st.source === 'series' ? currentSeries() : null;
      if (s && (force || !name.value || name.dataset.auto === name.value)) {
        name.value = slugify(s.title || '');
        name.dataset.auto = name.value;
      }
      renderForm();
    }

    function tabs(el, items, current, onPick) {
      el.innerHTML = '';
      items.forEach(it => {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = it.label;
        b.className = current === it.key ? 'active' : '';
        b.onclick = () => onPick(it.key);
        el.appendChild(b);
      });
    }

    function renderForm() {
      if (!st.lib) return;
      const opts = st.lib.options || {};
      const min = (opts.images && opts.images.min) || 3;
      const n = imageCount();

      tabs(q('.et-source'), [{ key: 'series', label: 'Bildserie' }, { key: 'images', label: 'Bilder wählen' }],
        st.source, (k) => { st.source = k; if (k === 'images' && !st.pickPage) loadPicks(true); suggestName(); });
      q('.et-series-field').style.display = st.source === 'series' ? '' : 'none';
      q('.et-pick-field').style.display = st.source === 'images' ? '' : 'none';
      q('.et-pick-count').textContent = st.picked.size ? `· ${st.picked.size} gewählt` : '';

      const s = currentSeries();
      q('.et-series-hint').textContent = !s ? ''
        : s.item_count < min ? `Mindestens ${min} Bilder nötig.`
        : s.item_count < 12 ? `${s.item_count} Bilder — kentskooking nimmt etwa 20, weniger geht aber.`
        : `${s.item_count} Bilder.`;

      const slug = slugify(q('.et-name').value);
      const taken = st.lib.trainings.some(t => t.name === slug);
      q('.et-name-hint').textContent = !slug ? '' : taken ? `„${slug}“ gibt es schon.` : `Im Render: artrium/${slug}`;

      tabs(q('.et-template'), TEMPLATES, st.template, (k) => { st.template = k; renderForm(); });
      q('.et-template-hint').textContent = (TEMPLATES.find(t => t.key === st.template) || {}).hint || '';

      q('.et-vectors-val').textContent = q('.et-vectors').value;
      const steps = parseInt(q('.et-steps').value, 10);
      const rate = opts.steps_per_second || 3.5;
      const every = opts.save_every || 250;
      const secs = 15 + steps / rate + (steps / every + 1) * 3 * 1.5;
      q('.et-eta').textContent = `≈ ${Math.round(secs / 60)} min auf der 4060 Ti (gemessen ${rate} Schritte/s) · `
        + `alle ${every} Schritte ein Snapshot mit Vorschau.`;

      const busy = st.lib.trainings.some(running);
      const btn = q('.et-start');
      btn.disabled = n < min || !slug || taken || busy;
      btn.textContent = busy ? 'Es läuft schon ein Training' : 'Training starten';
      q('.et-form-value').textContent = n ? `${n} Bilder · ${steps} Schritte` : '';
    }

    async function loadPicks(reset) {
      if (st.pickLoading || (!reset && st.pickDone)) return;
      if (reset) { st.pickPage = 0; st.pickDone = false; q('.et-pick-grid').innerHTML = ''; }
      st.pickLoading = true;
      const search = st.pickSearch ? `&search=${encodeURIComponent(st.pickSearch)}` : '';
      try {
        const imgs = await (await api(
          `/api/images?limit=${PICK_PAGE}&offset=${st.pickPage * PICK_PAGE}${search}`)).json();
        if (imgs.length < PICK_PAGE) st.pickDone = true;
        st.pickPage++;
        const grid = q('.et-pick-grid');
        for (const img of imgs) {
          const cell = document.createElement('div');
          cell.className = 'et-pick' + (st.picked.has(img.id) ? ' on' : '');
          cell.innerHTML = `<img src="${withAuth(img.thumb_url)}" alt="" loading="lazy">`;
          cell.onclick = () => {
            if (st.picked.has(img.id)) st.picked.delete(img.id); else st.picked.set(img.id, img);
            cell.classList.toggle('on', st.picked.has(img.id));
            renderForm();
          };
          grid.appendChild(cell);
        }
        q('.et-pick-more').style.display = st.pickDone ? 'none' : '';
      } catch {
        ArtRium.toast('Bilder konnten nicht geladen werden.', true);
      } finally {
        st.pickLoading = false;
      }
    }

    async function start() {
      const body = {
        name: q('.et-name').value,
        template: st.template,
        vectors: parseInt(q('.et-vectors').value, 10),
        steps: parseInt(q('.et-steps').value, 10),
        learning_rate: parseFloat(q('.et-lr').value) || null,
        init_word: (q('.et-init').value || '').trim() || null,
      };
      if (st.source === 'series') body.series_id = q('.et-series').value;
      else body.image_ids = [...st.picked.keys()];
      q('.et-start').disabled = true;
      try {
        const t = await (await api('/api/embeddings/trainings', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        })).json();
        ArtRium.toast(`Training „${t.name}“ gestartet — etwa ${Math.round(t.estimate_seconds / 60)} min.`);
        q('.et-form').open = false;
        st.picked.clear();
        root.querySelectorAll('.et-pick.on').forEach(c => c.classList.remove('on'));
        await reload();
      } catch (e) {
        ArtRium.toast(e.message || 'Training konnte nicht starten', true);
        renderForm();
      }
    }

    // ── Wiring ─────────────────────────────────────────────────────────────
    q('.et-series').onchange = () => suggestName(true);
    q('.et-name').oninput = renderForm;
    q('.et-vectors').oninput = renderForm;
    q('.et-steps').onchange = renderForm;
    q('.et-kent').onclick = () => { q('.et-vectors').value = '70'; renderForm(); };
    q('.et-start').onclick = start;
    q('.et-pick-more').onclick = () => loadPicks(false);
    let searchTimer;
    q('.et-pick-search').oninput = (e) => {
      clearTimeout(searchTimer);
      st.pickSearch = e.target.value;
      searchTimer = setTimeout(() => loadPicks(true), 350);
    };

    loadSeries();
    reload();
    return { reload, stop, get lib() { return st.lib; } };
  }

  // shared.js declares `const ArtRium` — a global binding every classic
  // script shares, but not a property of `window`. Extend that object.
  ArtRium.EmbeddingTrainer = { mount, slugify };
})();
