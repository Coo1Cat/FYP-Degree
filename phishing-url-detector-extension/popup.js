document.addEventListener('DOMContentLoaded', function() {
  console.log('Popup script loaded'); // Add this line to check if the script is loaded

  document.getElementById('checkUrlButton').addEventListener('click', async () => {
    const url = document.getElementById('urlInput').value;
    const resultElement = document.getElementById('result');

    if (url) {
      console.log(`Checking URL: ${url}`); // Log the URL being checked
      try {
        const response = await fetch('http://127.0.0.1:5000/check_url', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ url })
        });

        console.log('Response received:', response); // Log the response

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Data received:', data); // Log the response data

        if (data.isPhishing) {
          resultElement.textContent = 'The URL is a phishing site!';
          resultElement.style.color = 'red';
        } else {
          resultElement.textContent = 'The URL is Legit.';
          resultElement.style.color = 'green';
        }
      } catch (error) {
        console.error('Error checking URL:', error); // Log any errors
        resultElement.textContent = 'Error checking URL. Please try again.';
      }
    } else {
      resultElement.textContent = 'Please enter a URL.';
    }
  });
});
