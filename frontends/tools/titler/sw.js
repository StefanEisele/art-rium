importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-titler-v1',
  shell: [
    '/tools/titler/',
    '/tools/titler/manifest.json',
    '/tools/titler/icon.svg',
  ],
});
