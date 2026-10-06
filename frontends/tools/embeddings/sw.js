importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-embeddings-v1',
  shell: [
    '/tools/embeddings/',
    '/tools/embeddings/manifest.json',
    '/tools/embeddings/icon.svg',
  ],
  // /shared/ evolves in lockstep with every page's markup — never let it go
  // stale behind a cached copy.
  excludeFromCache: ['/shared/'],
});
