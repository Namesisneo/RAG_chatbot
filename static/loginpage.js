const users = {
    "bits": "1234",
    "mnit": "1234"
};

function validateLogin() {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    if (users[username] && users[username] === password) {
        const videoOverlay = document.getElementById("video-overlay");
        videoOverlay.style.display = "flex";
        const video = document.getElementById("overlay-video");
        video.play();

        // Stop the video after 5 seconds
        setTimeout(function() {
            video.pause();
            video.currentTime = 0;
            videoOverlay.style.display = "none";
            window.location.href = "/dashboard";
        }, 5000);

        video.addEventListener("ended", function() {
            window.location.href = "/dashboard";
        });
        return false;
    } else {
        alert("Invalid username or password.");
        return false;
    }
}