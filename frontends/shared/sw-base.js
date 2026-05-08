/**
 * art-rium shared service-worker logic.
 *
 * Each tool's sw.js is reduced to:
 *
 *   importScripts('/shared/sw-base.js');
 *   artRiumSetupSw({ cache: 'tool-name-v1', shell: ['/tools/x/', ...] });
 *
 * Optional `excludeFromCache` lets the dashboard sw skip /tools/* requests.
 */
self.artRiumSetupSw = ({ cache, shell, excludeFromCache = [] }) => {
  self.addEventListener('install', e => {
    e.waitUntil(
      caches.open(cache).then(c => c.addAll(shell)).then(() => self.skipWaiting())
    );
  });

  self.addEventListener('activate', e => {
    e.waitUntil(
      caches.keys().then(keys =>
        Promise.all(keys.filter(k => k !== cache).map(k => caches.delete(k)))
      ).then(() => self.clients.claim())
    );
  });

  self.addEventListener('fetch', e => {
    const url = e.request.url;
    // Never intercept API calls, WebSocket upgrades, or per-sw exclusions.
    if (url.includes('/api/') || url.includes('/ws/')) return;
    if (excludeFromCache.some(prefix => url.includes(prefix))) return;

    e.respondWith(
      caches.match(e.request).then(cached => {
        const network = fetch(e.request).then(res => {
          if (res.ok && e.request.method === 'GET') {
            const clone = res.clone();
            caches.open(cache).then(c => c.put(e.request, clone));
          }
          return res;
        });
        return cached || network;
      })
    );
  });
};
