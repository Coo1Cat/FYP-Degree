chrome.webRequest.onBeforeRequest.addListener(
  async function(details) {
    const url = details.url;

    const response = await fetch('http://localhost:5000/check_url', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url })
    });
    const data = await response.json();

    if (data.isPhishing) {
      // Redirect to a warning page or block the request
      return { redirectUrl: chrome.runtime.getURL('warning.html?url=' + encodeURIComponent(url)) };
    }
    return { cancel: false };
  },
  { urls: ["<all_urls>"] },
  ["blocking"]
);
