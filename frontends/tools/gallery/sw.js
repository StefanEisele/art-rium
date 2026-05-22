importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-gallery-v2',
  shell: [
    '/tools/gallery/',
    '/tools/gallery/manifest.json',
    '/tools/gallery/icon.svg',
  ],
});
