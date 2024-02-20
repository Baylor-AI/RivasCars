chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    if (request.type === "setCookie") {
        const { score, justification } = request.data;
        const data = JSON.stringify({ score, justification });
        chrome.cookies.set({
            url: "https://www.craigslist.org",
            name: "analysisResult",
            value: encodeURIComponent(data),
            expirationDate: (new Date().getTime()/1000) + 3600 // Expires in 1 hour
        });
    }
});
