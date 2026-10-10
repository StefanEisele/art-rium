importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-iterate-v2',
  shell: [
    '/tools/iterate/',
    '/tools/iterate/manifest.json',
    '/tools/iterate/icon.svg',
  ],
  // /shared/ evolves in lockstep with every page's markup — never let it go
  // stale behind a cached copy.
  excludeFromCache: ['/shared/'],
});
