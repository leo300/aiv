const CACHE = "ai-vision-v1";
self.addEventListener("install", event => {
    event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(["/", "/index.html", "/app.js", "/style.css", "/manifest.json"])));
});
self.addEventListener("fetch", event => {
    event.respondWith(caches.match(event.request).then(response => {
        return response || fetch(event.request);
    }));
});
