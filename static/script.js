/* =========================================
   SHOPASSISTAI RAG 2.0
   CHATBOT FRONTEND JAVASCRIPT
========================================= */


/* =========================================
   DOM ELEMENTS
========================================= */

const chatForm = document.getElementById("chatForm");

const messageInput = document.getElementById("messageInput");

const messagesArea = document.getElementById("messagesArea");

const welcomeArea = document.getElementById("welcomeArea");

const typingIndicator = document.getElementById("typingIndicator");

const sendButton = document.getElementById("sendButton");

const characterCount = document.getElementById("characterCount");

const clearChatBtn = document.getElementById("clearChatBtn");

const newChatBtn = document.getElementById("newChatBtn");

const toast = document.getElementById("toast");

const menuBtn = document.getElementById("menuBtn");

const menuToggle = document.getElementById("menuToggle");

const sidebar = document.getElementById("sidebar");


/* =========================================
   APPLICATION STATE
========================================= */

let isWaitingForResponse = false;

let toastTimer = null;


/* =========================================
   TOAST NOTIFICATION
========================================= */

function showToast(message, type = "success") {

    const toastMessage = toast.querySelector(".toast-message");

    toastMessage.textContent = message;

    toast.classList.remove("error");

    if (type === "error") {
        toast.classList.add("error");
    }

    toast.classList.add("show");

    clearTimeout(toastTimer);

    toastTimer = setTimeout(() => {

        toast.classList.remove("show");

    }, 3000);

}


/* =========================================
   GET CURRENT TIME
========================================= */

function getCurrentTime() {

    return new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });

}


/* =========================================
   ESCAPE HTML
========================================= */

function escapeHTML(text) {

    const element = document.createElement("div");

    element.textContent = String(text ?? "");

    return element.innerHTML;

}


/* =========================================
   BASIC MARKDOWN FORMATTER
========================================= */

function formatMessage(text) {

    let content = escapeHTML(text);

    // Code blocks
    content = content.replace(
        /```([\s\S]*?)```/g,
        "<pre><code>$1</code></pre>"
    );

    // Inline code
    content = content.replace(
        /`([^`]+)`/g,
        "<code>$1</code>"
    );

    // Bold text
    content = content.replace(
        /\*\*(.*?)\*\*/g,
        "<strong>$1</strong>"
    );

    // Italic text
    content = content.replace(
        /\*(.*?)\*/g,
        "<em>$1</em>"
    );

    // Convert URLs to clickable links
    content = content.replace(
        /(https?:\/\/[^\s<]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // Paragraphs and line breaks
    const blocks = content.split(/\n{2,}/);

    content = blocks.map(block => {

        block = block.trim();

        if (!block) {
            return "";
        }

        // Preserve code blocks
        if (block.startsWith("<pre>")) {
            return block;
        }

        // Unordered lists
        if (/^[-*]\s+/m.test(block)) {

            const items = block.split("\n").map(line => {

                const item = line.replace(/^[-*]\s+/, "").trim();

                return item ? `<li>${item}</li>` : "";

            }).join("");

            return `<ul>${items}</ul>`;
        }

        // Ordered lists
        if (/^\d+\.\s+/m.test(block)) {

            const items = block.split("\n").map(line => {

                const item = line.replace(/^\d+\.\s+/, "").trim();

                return item ? `<li>${item}</li>` : "";

            }).join("");

            return `<ol>${items}</ol>`;
        }

        return `<p>${block.replace(/\n/g, "<br>")}</p>`;

    }).join("");

    return content;

}


/* =========================================
   ADD MESSAGE TO CHAT
========================================= */

function addMessage(text, sender) {

    const message = document.createElement("div");

    const isUser = sender === "user";

    message.className = isUser
        ? "message user-message"
        : "message assistant-message";

    const avatar = isUser ? "U" : "✦";

    const senderName = isUser
        ? "You"
        : "ShopAssistAI";

    message.innerHTML = `

        <div class="message-avatar">
            ${avatar}
        </div>

        <div class="message-content">

            <div class="message-header">

                <strong>${senderName}</strong>

                <span class="message-time">
                    ${getCurrentTime()}
                </span>

            </div>

            <div class="message-bubble">
                ${formatMessage(text)}
            </div>

        </div>

    `;

    messagesArea.appendChild(message);

    scrollToBottom();

}


/* =========================================
   SCROLL TO LATEST MESSAGE
========================================= */

function scrollToBottom() {

    const chatSection = document.querySelector(".chat-section");

    requestAnimationFrame(() => {

        chatSection.scrollTo({
            top: chatSection.scrollHeight,
            behavior: "smooth"
        });

    });

}


/* =========================================
   SHOW / HIDE TYPING INDICATOR
========================================= */

function showTypingIndicator() {

    typingIndicator.classList.add("active");

    scrollToBottom();

}


function hideTypingIndicator() {

    typingIndicator.classList.remove("active");

}


/* =========================================
   UPDATE CHARACTER COUNT
========================================= */

function updateCharacterCount() {

    const currentLength = messageInput.value.length;

    characterCount.textContent = `${currentLength} / 2000`;

}


/* =========================================
   AUTO RESIZE TEXTAREA
========================================= */

function autoResizeTextarea() {

    messageInput.style.height = "auto";

    messageInput.style.height =
        Math.min(messageInput.scrollHeight, 130) + "px";

}


/* =========================================
   SET LOADING STATE
========================================= */

function setLoadingState(loading) {

    isWaitingForResponse = loading;

    sendButton.disabled = loading;

    messageInput.disabled = loading;

    sendButton.innerHTML = loading
        ? `
            <span>Thinking</span>
            <span>...</span>
          `
        : `
            <span>Send</span>
            <svg viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2">
                <path d="m22 2-7 20-4-9-9-4Z"/>
                <path d="M22 2 11 13"/>
            </svg>
          `;

}


/* =========================================
   SEND MESSAGE TO FLASK BACKEND
========================================= */

async function sendMessage(message) {

    if (isWaitingForResponse) {
        return;
    }

    const cleanMessage = message.trim();

    if (!cleanMessage) {
        return;
    }

    // Hide welcome screen
    welcomeArea.style.display = "none";

    // Display user message
    addMessage(cleanMessage, "user");

    // Clear input
    messageInput.value = "";

    updateCharacterCount();

    autoResizeTextarea();

    // Show loading animation
    setLoadingState(true);

    showTypingIndicator();

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: cleanMessage
            })

        });


        // Read server response
        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                data.message ||
                `Server error: ${response.status}`
            );

        }


        // Get AI answer
        const answer =
            data.answer ||
            data.response ||
            data.reply ||
            data.message ||
            data.result;


        if (!answer) {

            throw new Error(
                "The server returned an empty response."
            );

        }


        // Hide loading animation
        hideTypingIndicator();

        // Display AI response
        addMessage(answer, "assistant");


    } catch (error) {

        console.error("Chat API Error:", error);

        hideTypingIndicator();

        addMessage(
            "Sorry, I couldn't process your request right now. Please try again. " +
            "If the issue continues, check the Flask server logs.",
            "assistant"
        );

        showToast(
            error.message || "Something went wrong.",
            "error"
        );

    } finally {

        setLoadingState(false);

        messageInput.focus();

        scrollToBottom();

    }

}


/* =========================================
   CHAT FORM SUBMISSION
========================================= */

chatForm.addEventListener("submit", function(event) {

    event.preventDefault();

    if (isWaitingForResponse) {
        return;
    }

    const message = messageInput.value;

    sendMessage(message);

});


/* =========================================
   ENTER TO SEND MESSAGE
========================================= */

messageInput.addEventListener("keydown", function(event) {

    if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !event.isComposing
    ) {

        event.preventDefault();

        chatForm.requestSubmit();

    }

});


/* =========================================
   INPUT EVENTS
========================================= */

messageInput.addEventListener("input", function() {

    updateCharacterCount();

    autoResizeTextarea();

});


/* =========================================
   SUGGESTED QUESTION BUTTONS
========================================= */

const suggestionCards = document.querySelectorAll(
    ".suggestion-card"
);

suggestionCards.forEach(card => {

    card.addEventListener("click", function() {

        const prompt = this.dataset.prompt;

        if (!prompt || isWaitingForResponse) {
            return;
        }

        sendMessage(prompt);

    });

});


/* =========================================
   RESET CHAT
========================================= */

async function resetChat() {

    if (isWaitingForResponse) {
        return;
    }

    try {

        const response = await fetch("/reset", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            }

        });


        if (!response.ok) {

            let errorMessage = "Unable to reset conversation.";

            try {
                const data = await response.json();

                errorMessage = data.error || errorMessage;

            } catch (_) {
                // Keep the default error message.
            }

            throw new Error(errorMessage);

        }


        // Clear displayed messages
        messagesArea.innerHTML = "";

        // Show welcome screen again
        welcomeArea.style.display = "";

        // Clear input
        messageInput.value = "";

        updateCharacterCount();

        autoResizeTextarea();

        hideTypingIndicator();

        showToast("New conversation started.");

        messageInput.focus();

    } catch (error) {

        console.error("Reset Error:", error);

        showToast(
            error.message || "Unable to reset chat.",
            "error"
        );

    }

}


/* =========================================
   CLEAR CHAT BUTTON
========================================= */

clearChatBtn.addEventListener("click", function() {

    resetChat();

});


/* =========================================
   NEW CHAT BUTTON
========================================= */

newChatBtn.addEventListener("click", function() {

    resetChat();

    closeSidebar();

});


/* =========================================
   MOBILE SIDEBAR
========================================= */

function openSidebar() {

    sidebar.classList.add("open");

}


function closeSidebar() {

    sidebar.classList.remove("open");

}


if (menuBtn) {

    menuBtn.addEventListener("click", function() {

        closeSidebar();

    });

}


if (menuToggle) {

    menuToggle.addEventListener("click", function() {

        sidebar.classList.toggle("open");

    });

}


/* =========================================
   CLOSE SIDEBAR WHEN CLICKING OUTSIDE
========================================= */

document.addEventListener("click", function(event) {

    if (
        window.innerWidth <= 700 &&
        sidebar.classList.contains("open") &&
        !sidebar.contains(event.target) &&
        !menuToggle.contains(event.target)
    ) {

        closeSidebar();

    }

});


/* =========================================
   INITIALIZE APPLICATION
========================================= */

document.addEventListener("DOMContentLoaded", function() {

    updateCharacterCount();

    autoResizeTextarea();

    messageInput.focus();

    console.log(
        "ShopAssistAI RAG 2.0 Frontend Initialized Successfully."
    );

});
