function startVideo() {
    console.log("Starting video...");
    fetch('/start', { method: 'POST' })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.text();
        })
        .then(data => {
            console.log("Start response:", data);
            alert(data);
            const videoFeed = document.getElementById('video-feed');
            videoFeed.style.display = 'block';
            videoFeed.src = videoFeed.src;
        })
        .catch(error => {
            console.error('Error starting video:', error);
            alert('Failed to start video: ' + error);
        });
}

function pauseVideo() {
    console.log("Pausing video...");
    fetch('/pause', { method: 'POST' })
        .then(response => response.text())
        .then(data => {
            console.log("Pause response:", data);
            alert(data);
        })
        .catch(error => {
            console.error('Error pausing video:', error);
            alert('Failed to pause video: ' + error);
        });
}

function closeVideo() {
    console.log("Closing video...");
    fetch('/close', { method: 'POST' })
        .then(response => response.text())
        .then(data => {
            console.log("Close response:", data);
            alert(data);
            window.location.href = '/';
        })
        .catch(error => {
            console.error('Error closing video:', error);
            alert('Failed to close video: ' + error);
        });
}

document.addEventListener('DOMContentLoaded', function() {
    console.log("Page loaded, hiding video feed initially.");
    document.getElementById('video-feed').style.display = 'none';
});