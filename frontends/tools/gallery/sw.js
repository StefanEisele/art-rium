importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-gallery-v3',
  shell: [
    '/tools/gallery/',
    '/tools/gallery/manifest.json',
    '/tools/gallery/icon.svg',
  ],
  // /shared/ (shared.css/shared.js) evolves in lockstep with every page's
  // markup — never let it go stale behind a cached copy.
  excludeFromCache: ['/shared/'],
});
