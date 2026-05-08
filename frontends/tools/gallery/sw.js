importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-gallery-v1',
  shell: [
    '/tools/gallery/',
    '/tools/gallery/manifest.json',
    '/tools/gallery/icon.svg',
  ],
});
