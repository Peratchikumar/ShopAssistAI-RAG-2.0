/* ============================================================
   SHOPASSISTAI RAG 2.0
   Advanced E-Commerce Customer Support Chatbot
   File: static/script.js
   Backend: Flask
   ============================================================ */

"use strict";


/* ============================================================
   1. DOM ELEMENTS
   ============================================================ */

const chatArea = document.getElementById("chatArea");
const welcomeScreen = document.getElementById("welcomeScreen");
const messagesContainer = document.getElementById("messagesContainer");

const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");

const newChatBtn = document.getElementById("newChatBtn");
const mobileMenuBtn = document.getElementById("mobileMenuBtn");
const sidebar = document.getElementById("sidebar");

const suggestionCards = document.querySelectorAll(".suggestion-card");


/* ============================================================
   2. APPLICATION STATE
   ============================================================ */

let isSending = false;
let isResetting = false;
let typingElement = null;


/* ============================================================
   3. UTILITY FUNCTIONS
   ============================================================ */

function escapeHTML(value) {
    const div = document.createElement("div");
    div.textContent = String(value ?? "");
    return div.innerHTML;
}


function getCurrentTime() {
    return new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
    });
}


function scrollToBottom() {
    requestAnimationFrame(() => {
        chatArea.scrollTo({
            top: chatArea.scrollHeight,
            behavior: "smooth"
        });
    });
}


function toggleWelcomeScreen(show) {
    welcomeScreen.style.display = show ? "block" : "none";
}


function updateSendButton() {
    const hasMessage = messageInput.value.trim().length > 0;

    sendBtn.disabled = (
        !hasMessage ||
        isSending ||
        isResetting
    );
}


function updateNewChatButton() {
    newChatBtn.disabled = isSending || isResetting;
}


/* ============================================================
   4. SAFE MESSAGE FORMATTING
   ============================================================ */

/*
    Supports:
    - Bold text using **text**
    - Bullet lists
    - Numbered lists
    - Paragraphs

    All original text is escaped before formatting.
    Raw HTML from the AI is never executed.
*/

function formatMessage(text) {

    const escapedText = escapeHTML(text);

    const lines = escapedText.split(/\r?\n/);

    let html = "";
    let listType = null;

    function closeList() {
        if (listType) {
            html += `</${listType}>`;
            listType = null;
        }
    }

    function formatInline(line) {
        return line.replace(
            /\*\*(.+?)\*\*/g,
            "<strong>$1</strong>"
        );
    }

    for (const originalLine of lines) {

        const line = originalLine.trim();

        if (!line) {
            closeList();
            continue;
        }

        const bulletMatch = line.match(/^[-*]\s+(.+)$/);

        const numberMatch = line.match(/^\d+[.)]\s+(.+)$/);

        if (bulletMatch) {

            if (listType !== "ul") {
                closeList();
                html += "<ul>";
                listType = "ul";
            }

            html += `<li>${formatInline(bulletMatch[1])}</li>`;

            continue;
        }

        if (numberMatch) {

            if (listType !== "ol") {
                closeList();
                html += "<ol>";
                listType = "ol";
            }

            html += `<li>${formatInline(numberMatch[1])}</li>`;

            continue;
        }

        closeList();

        html += `<p>${formatInline(line)}</p>`;
    }

    closeList();

    return html || "<p>No response was returned.</p>";
}


/* ============================================================
   5. ADD USER MESSAGE
   ============================================================ */

function addUserMessage(message) {

    const messageElement = document.createElement("div");

    messageElement.className = "message user-message";

    messageElement.innerHTML = `
        <div class="message-content">

            <div class="message-meta">
                <strong>You</strong>
                <span>${getCurrentTime()}</span>
            </div>

            <div class="message-bubble">
                ${escapeHTML(message).replace(/\r?\n/g, "<br>")}
            </div>

        </div>

        <div class="message-avatar">
            <i class="fa-solid fa-user"></i>
        </div>
    `;

    messagesContainer.appendChild(messageElement);

    scrollToBottom();
}


/* ============================================================
   6. SOURCE DOCUMENT FORMATTING
   ============================================================ */

function buildSourcesHTML(sources) {

    if (!Array.isArray(sources) || sources.length === 0) {
        return "";
    }

    const uniqueSources = new Map();

    sources.forEach(source => {

        let sourceName = "";
        let page = "";

        if (typeof source === "string") {
            sourceName = source;
        } else if (source && typeof source === "object") {

            sourceName = (
                source.source ||
                source.filename ||
                source.file_name ||
                source.document ||
                "Unknown source"
            );

            page = source.page
                ? `Page ${source.page}`
                : "";
        }

        if (!sourceName) {
            return;
        }

        const key = `${sourceName}-${page}`;

        uniqueSources.set(key, {
            name: sourceName,
            page: page
        });

    });

    if (uniqueSources.size === 0) {
        return "";
    }

    const sourceItems = Array.from(uniqueSources.values())
        .map(source => `
            <div class="source-item">
                <i class="fa-solid fa-file-pdf"></i>
                <span>${escapeHTML(source.name)}</span>
                ${
                    source.page
                        ? `<small>${escapeHTML(source.page)}</small>`
                        : ""
                }
            </div>
        `)
        .join("");

    return `
        <div class="source-container">

            <div class="source-title">
                <i class="fa-solid fa-book-open"></i>
                Knowledge Base Sources
            </div>

            ${sourceItems}

        </div>
    `;
}


/* ============================================================
   7. ADD AI MESSAGE
   ============================================================ */

function addAIMessage(answer, sources = []) {

    const messageElement = document.createElement("div");

    messageElement.className = "message ai-message";

    const formattedAnswer = formatMessage(answer);

    const sourcesHTML = buildSourcesHTML(sources);

    messageElement.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid fa-robot"></i>
        </div>

        <div class="message-content">

            <div class="message-meta">
                <strong>ShopAssistAI</strong>
                <span>${getCurrentTime()}</span>
            </div>

            <div class="message-bubble">
                ${formattedAnswer}
            </div>

            ${sourcesHTML}

        </div>
    `;

    messagesContainer.appendChild(messageElement);

    scrollToBottom();
}


/* ============================================================
   8. TYPING INDICATOR
   ============================================================ */

function showTypingIndicator() {

    removeTypingIndicator();

    typingElement = document.createElement("div");

    typingElement.className = "message ai-message";

    typingElement.innerHTML = `
        <div class="message-avatar">
            <i class="fa-solid fa-robot"></i>
        </div>

        <div class="message-content">

            <div class="message-meta">
                <strong>ShopAssistAI</strong>
                <span>Thinking...</span>
            </div>

            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>

        </div>
    `;

    messagesContainer.appendChild(typingElement);

    scrollToBottom();
}


function removeTypingIndicator() {

    if (typingElement) {
        typingElement.remove();
        typingElement = null;
    }
}


/* ============================================================
   9. API RESPONSE HANDLING
   ============================================================ */

async function readJSONResponse(response) {

    const contentType = response.headers.get("content-type") || "";

    if (!contentType.includes("application/json")) {
        throw new Error(
            `The server returned an unexpected response (${response.status}).`
        );
    }

    return await response.json();
}


/* ============================================================
   10. SEND MESSAGE TO FLASK BACKEND
   ============================================================ */

async function sendMessage(customMessage = null) {

    if (isSending || isResetting) {
        return;
    }

    const message = (
        customMessage !== null
            ? customMessage
            : messageInput.value
    ).trim();

    if (!message) {
        return;
    }

    if (message.length > 2000) {

        addAIMessage(
            "Your message is too long. Please keep it within 2,000 characters."
        );

        return;
    }

    isSending = true;

    updateSendButton();
    updateNewChatButton();

    // Hide welcome screen.
    toggleWelcomeScreen(false);

    // Display user message.
    addUserMessage(message);

    // Clear input.
    messageInput.value = "";
    messageInput.style.height = "auto";

    // Show typing animation.
    showTypingIndicator();

    try {

        const response = await fetch("/api/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },

            credentials: "same-origin",

            body: JSON.stringify({
                message: message
            })

        });

        const data = await readJSONResponse(response);

        removeTypingIndicator();

        // Handle unsuccessful HTTP response.
        if (!response.ok) {

            addAIMessage(
                data.message ||
                data.error ||
                `Server error (${response.status}). Please try again.`
            );

            return;
        }

        // Handle application-level error.
        if (
            !data.status ||
            String(data.status).toLowerCase() !== "success"
        ) {

            addAIMessage(
                data.message ||
                data.error ||
                "The chatbot could not process your request."
            );

            return;
        }

        // Extract generated answer.
        const answer = (
            typeof data.answer === "string" &&
            data.answer.trim()
        )
            ? data.answer
            : "I could not generate an answer. Please try again.";

        // Extract source documents.
        const sources = Array.isArray(data.sources)
            ? data.sources
            : [];

        // Display response.
        addAIMessage(answer, sources);

        console.log("ShopAssistAI response:", {
            responseTime: data.response_time,
            tokenUsage: data.token_usage,
            sourceCount: sources.length
        });

    }

    catch (error) {

        console.error("ShopAssistAI chat error:", error);

        removeTypingIndicator();

        addAIMessage(
            "I'm unable to connect to the chatbot service. " +
            "Please check your server and try again."
        );

    }

    finally {

        isSending = false;

        updateSendButton();
        updateNewChatButton();

        messageInput.focus();

    }
}


/* ============================================================
   11. RESET CONVERSATION
   ============================================================ */

async function resetConversation() {

    if (isSending || isResetting) {
        return;
    }

    isResetting = true;

    updateSendButton();
    updateNewChatButton();

    try {

        const response = await fetch("/api/reset", {

            method: "POST",

            headers: {
                "Accept": "application/json"
            },

            credentials: "same-origin"

        });

        const data = await readJSONResponse(response);

        if (
            !response.ok ||
            !data.status ||
            String(data.status).toLowerCase() !== "success"
        ) {

            throw new Error(
                data.message ||
                data.error ||
                "Unable to reset the conversation."
            );

        }

        // Clear conversation messages.
        messagesContainer.innerHTML = "";

        // Remove typing indicator.
        removeTypingIndicator();

        // Restore welcome screen.
        toggleWelcomeScreen(true);

        // Reset input.
        messageInput.value = "";
        messageInput.style.height = "auto";

        // Close mobile sidebar.
        sidebar.classList.remove("open");

        console.log("ShopAssistAI conversation reset successfully.");

    }

    catch (error) {

        console.error("ShopAssistAI reset error:", error);

        addAIMessage(
            "I couldn't reset the conversation. " +
            "Please try again."
        );

    }

    finally {

        isResetting = false;

        updateSendButton();
        updateNewChatButton();

        messageInput.focus();

    }
}


/* ============================================================
   12. SUGGESTED QUESTIONS
   ============================================================ */

suggestionCards.forEach(card => {

    card.addEventListener("click", () => {

        const question = card.dataset.question;

        if (question) {
            sendMessage(question);
        }

    });

});


/* ============================================================
   13. SEND BUTTON
   ============================================================ */

sendBtn.addEventListener("click", () => {
    sendMessage();
});


/* ============================================================
   14. ENTER KEY TO SEND
   ============================================================ */

messageInput.addEventListener("keydown", event => {

    if (event.key === "Enter" && !event.shiftKey) {

        event.preventDefault();

        sendMessage();

    }

});


/* ============================================================
   15. AUTO-RESIZE TEXTAREA
   ============================================================ */

messageInput.addEventListener("input", () => {

    messageInput.style.height = "auto";

    messageInput.style.height =
        Math.min(messageInput.scrollHeight, 130) + "px";

    updateSendButton();

});


/* ============================================================
   16. NEW CONVERSATION BUTTON
   ============================================================ */

newChatBtn.addEventListener("click", () => {
    resetConversation();
});


/* ============================================================
   17. MOBILE SIDEBAR
   ============================================================ */

mobileMenuBtn.addEventListener("click", event => {

    event.stopPropagation();

    sidebar.classList.toggle("open");

});


document.addEventListener("click", event => {

    const clickedInsideSidebar = sidebar.contains(event.target);

    const clickedMenuButton = mobileMenuBtn.contains(event.target);

    if (
        window.innerWidth <= 768 &&
        !clickedInsideSidebar &&
        !clickedMenuButton
    ) {
        sidebar.classList.remove("open");
    }

});


/* ============================================================
   18. INITIALIZE APPLICATION
   ============================================================ */

function initializeShopAssistAI() {

    console.log("=========================================");
    console.log("ShopAssistAI RAG 2.0");
    console.log("Advanced E-Commerce Customer Support");
    console.log("Frontend initialized successfully.");
    console.log("=========================================");

    toggleWelcomeScreen(true);

    updateSendButton();
    updateNewChatButton();

}


document.addEventListener(
    "DOMContentLoaded",
    initializeShopAssistAI
);
