importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-forms-v1',
  shell: [
    '/tools/forms/',
    '/tools/forms/manifest.json',
    '/tools/forms/icon.svg',
  ],
  // /shared/ evolves in lockstep with every page's markup — never let it go
  // stale behind a cached copy.
  excludeFromCache: ['/shared/'],
});
