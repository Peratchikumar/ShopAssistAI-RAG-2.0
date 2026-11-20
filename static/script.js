
/* =========================================
   SHOPASSISTAI - CHAT APPLICATION
   Flask API integration
   ========================================= */

document.addEventListener("DOMContentLoaded", () => {
    // DOM elements
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
    const sidebar = document.getElementById("sidebar");

    let isSending = false;
    let toastTimeout;

    // -------------------------------------
    // Utility functions
    // -------------------------------------

    function showToast(message) {
        toast.textContent = message;
        toast.classList.add("show");

        clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            toast.classList.remove("show");
        }, 2800);
    }

    function getCurrentTime() {
        return new Intl.DateTimeFormat([], {
            hour: "2-digit",
            minute: "2-digit"
        }).format(new Date());
    }

    function escapeHTML(value) {
        return String(value).replace(/[&<>"']/g, (character) => {
            const entities = {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#039;"
            };
            return entities[character];
        });
    }

    // Basic safe formatting for common Markdown in AI responses.
    function formatMessage(text) {
        let safeText = escapeHTML(text);

        safeText = safeText.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );

        safeText = safeText.replace(
            /`([^`]+)`/g,
            "<code>$1</code>"
        );

        safeText = safeText.replace(
            /^[-*]\s+(.+)$/gm,
            "• $1"
        );

        return safeText.replace(/\n/g, "<br>");
    }

    function scrollToBottom() {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }

    function hideWelcomeScreen() {
        welcomeArea.classList.add("hidden");
    }

    function showTyping() {
        typingIndicator.classList.remove("hidden");
        scrollToBottom();
    }

    function hideTyping() {
        typingIndicator.classList.add("hidden");
    }

    // -------------------------------------
    // Add chat messages
    // -------------------------------------

    function addMessage(text, sender) {
        const isUser = sender === "user";

        const messageRow = document.createElement("div");
        messageRow.className = `message-row ${isUser ? "user-message" : "assistant-message"}`;

        const avatar = document.createElement("div");
        avatar.className = `message-avatar ${isUser ? "user-small" : "assistant-small"}`;
        avatar.textContent = isUser ? "You" : "✦";

        const messageContent = document.createElement("div");
        messageContent.className = "message-content";

        const messageName = document.createElement("span");
        messageName.className = "message-name";
        messageName.textContent = isUser ? "You" : "ShopAssistAI";

        const messageBubble = document.createElement("div");
        messageBubble.className = "message-bubble";

        // Only escaped text and controlled formatting are inserted.
        messageBubble.innerHTML = formatMessage(text);

        const messageTime = document.createElement("span");
        messageTime.className = "message-time";
        messageTime.textContent = getCurrentTime();

        messageContent.appendChild(messageName);
        messageContent.appendChild(messageBubble);
        messageContent.appendChild(messageTime);

        if (isUser) {
            messageRow.appendChild(messageContent);
            messageRow.appendChild(avatar);
        } else {
            messageRow.appendChild(avatar);
            messageRow.appendChild(messageContent);
        }

        messagesArea.appendChild(messageRow);
        scrollToBottom();
    }

    // -------------------------------------
    // Extract assistant answer from API
    // -------------------------------------

    function getAssistantAnswer(data) {
        if (typeof data === "string") {
            return data;
        }

        if (!data || typeof data !== "object") {
            return "";
        }

        const possibleAnswers = [
            data.answer,
            data.response,
            data.reply,
            data.message,
            data.result
        ];

        for (const value of possibleAnswers) {
            if (typeof value === "string" && value.trim()) {
                return value.trim();
            }
        }

        return "";
    }

    // -------------------------------------
    // Send message to Flask backend
    // -------------------------------------

    async function sendMessage(message) {
        const cleanMessage = message.trim();

        if (!cleanMessage || isSending) {
            return;
        }

        hideWelcomeScreen();
        addMessage(cleanMessage, "user");

        messageInput.value = "";
        updateCharacterCount();
        autoResizeTextarea();

        isSending = true;
        sendButton.disabled = true;
        showTyping();

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

            const data = await response.json().catch(() => ({}));

            if (!response.ok) {
                throw new Error(
                    data.error ||
                    data.message ||
                    `Server error (${response.status}). Please try again.`
                );
            }

            const answer = getAssistantAnswer(data);

            if (!answer) {
                throw new Error("The server returned an empty response. Please try again.");
            }

            addMessage(answer, "assistant");

        } catch (error) {
            console.error("Chat API error:", error);

            addMessage(
                `Sorry, I couldn't process your request. ${error.message || "Please check your connection and try again."}`,
                "assistant"
            );

            showToast("Unable to get a response from the server.");
        } finally {
            hideTyping();
            isSending = false;
            sendButton.disabled = false;
            messageInput.focus();
        }
    }

    // -------------------------------------
    // Form submission
    // -------------------------------------

    chatForm.addEventListener("submit", (event) => {
        event.preventDefault();

        const message = messageInput.value;
        sendMessage(message);
    });

    // Enter sends, Shift + Enter creates a new line.
    messageInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();

            if (!isSending) {
                chatForm.requestSubmit();
            }
        }
    });

    // -------------------------------------
    // Textarea auto-resize
    // -------------------------------------

    function autoResizeTextarea() {
        messageInput.style.height = "auto";
        messageInput.style.height =
            `${Math.min(messageInput.scrollHeight, 120)}px`;
    }

    function updateCharacterCount() {
        characterCount.textContent =
            `${messageInput.value.length} / 2000`;
    }

    messageInput.addEventListener("input", () => {
        autoResizeTextarea();
        updateCharacterCount();
    });

    // -------------------------------------
    // Suggested question cards
    // -------------------------------------

    document.querySelectorAll(".suggestion-card").forEach((card) => {
        card.addEventListener("click", () => {
            const question = card.dataset.question;

            if (question && !isSending) {
                sendMessage(question);
            }
        });
    });

    // -------------------------------------
    // Clear / new conversation
    // -------------------------------------

    async function resetConversation() {
        if (isSending) {
            showToast("Please wait until the current response finishes.");
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
                throw new Error("Could not reset the server conversation.");
            }

            messagesArea.innerHTML = "";
            welcomeArea.classList.remove("hidden");
            messageInput.value = "";
            updateCharacterCount();
            autoResizeTextarea();

            showToast("Conversation cleared successfully.");
            messageInput.focus();

        } catch (error) {
            console.error("Reset error:", error);
            showToast(error.message || "Unable to clear the conversation.");
        }
    }

    clearChatBtn.addEventListener("click", resetConversation);
    newChatBtn.addEventListener("click", resetConversation);

    // -------------------------------------
    // Mobile sidebar
    // -------------------------------------

    menuBtn.addEventListener("click", () => {
        sidebar.classList.toggle("open");
    });

    document.addEventListener("click", (event) => {
        const clickedInsideSidebar = sidebar.contains(event.target);
        const clickedMenuButton = menuBtn.contains(event.target);

        if (
            window.innerWidth <= 680 &&
            sidebar.classList.contains("open") &&
            !clickedInsideSidebar &&
            !clickedMenuButton
        ) {
            sidebar.classList.remove("open");
        }
    });

    // -------------------------------------
    // Initial setup
    // -------------------------------------

    updateCharacterCount();
    autoResizeTextarea();
    messageInput.focus();
});
