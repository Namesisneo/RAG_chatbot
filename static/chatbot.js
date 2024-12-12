const chatbotIcon = document.getElementById("chatbot-icon");
const chatbotContainer = document.getElementById("chatbot-container");
const chatInput = document.getElementById("chat-input");
const chatBody = document.getElementById("chatbody");
const queryButtons = document.querySelector(".query-buttons");
const listeningIndicator = document.getElementById("listening-indicator");

function toggleChatbot() {
    if (chatbotContainer.style.display === "none" || chatbotContainer.style.display === "") {
        chatbotContainer.style.display = "flex";
        chatbotIcon.style.display = "none";
    } else {
        chatbotContainer.style.display = "none";
        chatbotIcon.style.display = "flex";
    }
}
function sendMessage() {
    const message = chatInput.value.trim();

    // Check if the message is empty
    if (message) {
        queryButtons.style.display = "none";

        // Display user message in chat body
        const userMessage = document.createElement("div");
        userMessage.className = "user-message";
        userMessage.textContent = message;
        chatBody.appendChild(userMessage);
        chatInput.value = "";

        chatBody.scrollTop = chatBody.scrollHeight;

        // Display typing indicator for bot
        const typingIndicator = document.createElement("div");
        typingIndicator.className = "bot-message typing-indicator";
        typingIndicator.innerHTML = '<span></span><span></span><span></span>';
        chatBody.appendChild(typingIndicator);

        // Log the message being sent
        console.log("Sending message:", message);

        // Send the message to the backend using fetch
            fetch("http://127.0.0.1:5000/chatbot", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message }),
        })
            .then((response) => {
                // Check if the response is ok (status code 200-299)
                if (!response.ok) {
                    return response.json().then((error) => {
                        throw new Error(error.error || "Unknown server error");
                    });
                }
                return response.json();
            })
            .then((data) => {
                // Remove typing indicator once the response is received
                typingIndicator.remove();

                // Display the bot's response in the chat
                const botMessage = document.createElement("div");
                botMessage.className = "bot-message";
                botMessage.textContent = data.reply || "Sorry, I couldn't process your message.";
                chatBody.appendChild(botMessage);

                chatBody.scrollTop = chatBody.scrollHeight;
            })
            .catch((error) => {
                // Remove typing indicator and display error message
                typingIndicator.remove();

                // Log the error details for debugging
                // console.error("Error sending message:", error);

                const errorMessage = document.createElement("div");
                errorMessage.className = "bot-message error-message";
                errorMessage.textContent = `Error: ${error.message || "Unknown error"}. Please try again later.`;
                chatBody.appendChild(errorMessage);

                chatBody.scrollTop = chatBody.scrollHeight;
            });
    } else {
        // If no message is entered, display an error message
        const errorMessage = document.createElement("div");
        errorMessage.className = "bot-message error-message";
        errorMessage.textContent = "Please enter a valid message.";
        chatBody.appendChild(errorMessage);
        chatBody.scrollTop = chatBody.scrollHeight;
    }
}



function startSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.onstart = function () {
        console.log("Speech recognition started");
        listeningIndicator.style.display = "block";
    };

    recognition.onresult = function (event) {
        const last = event.results.length - 1;
        const text = event.results[last][0].transcript;
        chatInput.value = text;
        sendMessage();
    };

    recognition.onerror = function (event) {
        console.error("Speech recognition error: " + event.error);
    };

    recognition.onend = function () {
        console.log("Speech recognition ended");
        listeningIndicator.style.display = "none";
    };

    recognition.start();
}
