importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'z-image-v4',
  shell: [
    '/tools/z-image/',
    '/tools/z-image/manifest.json',
    '/tools/z-image/icon.svg',
  ],
  // /shared/ (shared.css/shared.js) evolves in lockstep with every page's
  // markup — never let it go stale behind a cached copy.
  excludeFromCache: ['/shared/'],
});
