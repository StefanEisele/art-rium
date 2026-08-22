importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-vace-v1',
  shell: [
    '/tools/vace/',
    '/tools/vace/manifest.json',
    '/tools/vace/icon.svg',
  ],
  // /shared/ evolves in lockstep with every page's markup — never let it go
  // stale behind a cached copy.
  excludeFromCache: ['/shared/'],
});
