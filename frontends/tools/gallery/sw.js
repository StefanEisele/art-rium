importScripts('/shared/sw-base.js');

artRiumSetupSw({
  // Bumped with the network-first HTML change in sw-base.js: `activate` drops
  // every cache that is not this one, which is what evicts the stale page a
  // cache-first sw could otherwise serve forever.
  cache: 'art-rium-gallery-v7',
  shell: [
    '/tools/gallery/',
    '/tools/gallery/manifest.json',
    '/tools/gallery/icon.svg',
  ],
  // /shared/ (shared.css/shared.js) evolves in lockstep with every page's
  // markup — never let it go stale behind a cached copy.
  excludeFromCache: ['/shared/'],
});
