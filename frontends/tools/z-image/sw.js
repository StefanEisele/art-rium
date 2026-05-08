importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'z-image-v2',
  shell: [
    '/tools/z-image/',
    '/tools/z-image/manifest.json',
    '/tools/z-image/icon.svg',
  ],
});
