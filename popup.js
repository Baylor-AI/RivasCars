document.addEventListener('DOMContentLoaded', function() {
    chrome.storage.local.get(['analysisResult'], function(result) {
        if (result.analysisResult) {
            document.getElementById('result').innerHTML = `
                <p>Score: ${result.analysisResult.score}</p>
                <p>Justification: ${result.analysisResult.justification}</p>
            `;
        } else {
            document.getElementById('result').innerHTML = '<p>No results found.</p>';
        }
    });
});
