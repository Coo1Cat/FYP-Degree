chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getUrl") {
    sendResponse({ url: window.location.href });
  }
});
