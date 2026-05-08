importScripts('/shared/sw-base.js');

artRiumSetupSw({
  cache: 'art-rium-dashboard-v1',
  shell: ['/', '/manifest.json', '/icon.svg'],
  excludeFromCache: ['/tools/'],
});
