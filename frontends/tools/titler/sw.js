importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-titler-v3',
  shell: [
    '/tools/titler/',
    '/tools/titler/manifest.json',
    '/tools/titler/icon.svg',
  ],
  // /shared/ (shared.css/shared.js) evolves in lockstep with every page's
  // markup — never let it go stale behind a cached copy.
  excludeFromCache: ['/shared/'],
});
