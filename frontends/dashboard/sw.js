importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-dashboard-v2',
  shell: ['/', '/manifest.json', '/icon.svg'],
  // /shared/ (shared.css/shared.js) evolves in lockstep with every page's
  // markup — a stale cached copy silently breaks layout on every tool that
  // doesn't register its own service worker (this SW's scope is "/", so it
  // controls all of them). Never cache it, same as /tools/.
  excludeFromCache: ['/tools/', '/shared/'],
});
