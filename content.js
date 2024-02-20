function readCraigslistPost() {
  // Define selectors based on the Craigslist post structure
  const priceSelector = '.price';
  const titleSelector = '#titletextonly';
  const descriptionSelector = '#postingbody';
  const imageSelector = '.gallery img'; // This is a hypothetical selector; adjust based on actual structure

  // Extract data using the selectors
  const price = document.querySelector(priceSelector)?.textContent || 'No price found';
  const title = document.querySelector(titleSelector)?.textContent || 'No title found';
  const description = document.querySelector(descriptionSelector)?.textContent || 'No description found';
  const grabbedImages = [...document.querySelectorAll(imageSelector)].map(img => img.src);
  
  // Data object to send
  const postData = {
    title: title,
    price: price,
    description: description,
	images: grabbedImages
  };


  // Send data to the API
  sendDataToAPI(postData);
}

function sendDataToAPI(data) {
  fetch('https://172.111.50.46:5000/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  })
  .then(response => response.json())
  .then(data => {
	const analysisResult = { score: data.score, justification: data.justification }; // Replace with actual results
	chrome.storage.local.set({analysisResult: analysisResult}, function() {
		console.log('Analysis result is saved');
	});


    console.log('Success:', data);
  })
  .catch((error) => {
    console.error('Error:', error);
  });
}

// Run the function when the page is loaded
readCraigslistPost();
