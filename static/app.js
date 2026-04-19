/**
 * app.js — MediBot Frontend Logic
 * ----------------------------------
 * Handles:
 *  - Sending messages to the Flask API
 *  - Rendering user & bot messages with animations
 *  - Typing indicator, confidence bar, intent badges
 *  - Quick-example chips
 *  - Sidebar toggle (mobile)
 *  - Auto-resizing textarea
 *  - Keyboard shortcuts
 */

"use strict";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const chatWindow     = document.getElementById("chatWindow");
const userInput      = document.getElementById("userInput");
const sendBtn        = document.getElementById("sendBtn");
const clearBtn       = document.getElementById("clearBtn");
const intentList     = document.getElementById("intentList");
const welcomeCard    = document.getElementById("welcomeCard");
const sidebarToggle  = document.getElementById("sidebarToggle");
const sidebar        = document.querySelector(".sidebar");
const statusDot      = document.getElementById("statusDot");
const msgTemplate    = document.getElementById("msgTemplate");
const typingTemplate = document.getElementById("typingTemplate");

// ── State ─────────────────────────────────────────────────────────────────────
const API_BASE   = window.location.origin;
let   isLoading  = false;
let   msgCount   = 0;

// ── Intent icons ──────────────────────────────────────────────────────────────
const INTENT_ICONS = {
  allergy:      "🤧",
  anemia:       "🩸",
  arthritis:    "🦴",
  cold:         "🤒",
  diabetes:     "🍬",
  flu:          "🌡️",
  hypertension: "❤️",
  migraine:     "🧠",
  pneumonia:    "🫁",
};
const DEFAULT_ICON = "🩺";

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Format timestamp → "HH:MM" */
function fmtTime(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit",
  });
}

/** Escape HTML (for user messages) */
function escHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * Very lightweight markdown: **bold**, *italic*, newlines → <br>
 * Used only for bot bubbles where content is trusted.
 */
function renderMd(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n/g, "<br>");
}

/** Scroll chat to bottom smoothly */
function scrollToBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: "smooth" });
}

/** Hide welcome card once first message is sent */
function hideWelcome() {
  if (welcomeCard && welcomeCard.parentNode) {
    welcomeCard.style.animation = "fadeSlideUp 0.3s ease reverse forwards";
    setTimeout(() => welcomeCard.remove(), 300);
  }
}

/** Highlight the active intent in the sidebar */
function highlightIntent(intent) {
  document.querySelectorAll(".intent-chip").forEach((chip) => {
    chip.classList.toggle(
      "active-intent",
      chip.dataset.intent === intent
    );
  });
}

// ── Message rendering ─────────────────────────────────────────────────────────

/**
 * Append a user message bubble to the chat.
 */
function appendUserMessage(text) {
  msgCount++;
  const node  = msgTemplate.content.cloneNode(true);
  const wrap  = node.querySelector(".message");
  const avatar = node.querySelector(".avatar");
  const bubble = node.querySelector(".bubble");
  const meta   = node.querySelector(".meta");

  wrap.classList.add("user-message");
  avatar.classList.add("user-avatar");
  avatar.textContent = "You";
  bubble.classList.add("user-bubble");
  bubble.textContent = text;
  meta.textContent   = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  chatWindow.appendChild(node);
  scrollToBottom();
}

/**
 * Append a bot response bubble with intent badge + confidence bar.
 */
function appendBotMessage(data) {
  const { intent, confidence, response, timestamp } = data;
  const icon = INTENT_ICONS[intent] || DEFAULT_ICON;

  const node   = msgTemplate.content.cloneNode(true);
  const wrap   = node.querySelector(".message");
  const avatar = node.querySelector(".avatar");
  const bubble = node.querySelector(".bubble");
  const meta   = node.querySelector(".meta");

  wrap.classList.add("bot-message");
  avatar.classList.add("bot-avatar");
  avatar.textContent = icon;

  bubble.classList.add("bot-bubble");
  bubble.innerHTML   = renderMd(response);

  // Confidence bar
  const confPct = Math.round(confidence * 100);
  const confHtml = `
    <div class="confidence-bar-wrap">
      <span class="confidence-label">Confidence</span>
      <div class="confidence-track">
        <div class="confidence-fill" data-pct="${confPct}"></div>
      </div>
      <span class="confidence-val">${confPct}%</span>
    </div>`;
  bubble.insertAdjacentHTML("beforeend", confHtml);

  // Meta row
  meta.innerHTML = `
    <span class="intent-badge">${icon} ${intent}</span>
    <span>${fmtTime(timestamp)}</span>`;

  chatWindow.appendChild(node);

  // Animate the confidence bar after render
  requestAnimationFrame(() => {
    const fill = chatWindow.querySelector(
      `.confidence-fill[data-pct="${confPct}"]`
    );
    if (fill) {
      setTimeout(() => {
        fill.style.width = `${confPct}%`;
      }, 120);
    }
  });

  scrollToBottom();
  highlightIntent(intent);
}

/**
 * Append an error bubble.
 */
function appendErrorMessage(msg) {
  const node   = msgTemplate.content.cloneNode(true);
  const wrap   = node.querySelector(".message");
  const avatar = node.querySelector(".avatar");
  const bubble = node.querySelector(".bubble");
  const meta   = node.querySelector(".meta");

  wrap.classList.add("bot-message");
  avatar.classList.add("bot-avatar");
  avatar.textContent = "⚠️";
  bubble.classList.add("bot-bubble", "error-bubble");
  bubble.textContent = msg;
  meta.textContent   = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  chatWindow.appendChild(node);
  scrollToBottom();
}

// ── Typing indicator ──────────────────────────────────────────────────────────

function showTyping() {
  const node = typingTemplate.content.cloneNode(true);
  chatWindow.appendChild(node);
  scrollToBottom();
}

function hideTyping() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

// ── Send message flow ─────────────────────────────────────────────────────────

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isLoading) return;

  // UI state
  isLoading = true;
  sendBtn.disabled = true;
  userInput.value = "";
  userInput.style.height = "auto";
  statusDot.classList.add("loading-dot");
  hideWelcome();

  appendUserMessage(text);
  showTyping();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error: ${res.status}`);
    }

    const data = await res.json();
    hideTyping();
    appendBotMessage(data);

  } catch (err) {
    hideTyping();
    appendErrorMessage(
      `❌ ${err.message || "Could not reach the server. Make sure the Flask app is running."}`
    );
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    statusDot.classList.remove("loading-dot");
    userInput.focus();
  }
}

// ── Load intent list from API ─────────────────────────────────────────────────

async function loadIntents() {
  try {
    const res  = await fetch(`${API_BASE}/api/intents`);
    const data = await res.json();

    intentList.innerHTML = "";
    data.intents.forEach((intent) => {
      const li       = document.createElement("li");
      li.className   = "intent-chip";
      li.dataset.intent = intent;
      const icon     = INTENT_ICONS[intent] || DEFAULT_ICON;
      li.textContent = `${icon}  ${intent.charAt(0).toUpperCase() + intent.slice(1)}`;
      intentList.appendChild(li);
    });
  } catch {
    intentList.innerHTML =
      `<li class="intent-chip loading">Server offline</li>`;
  }
}

// ── Clear conversation ────────────────────────────────────────────────────────

function clearConversation() {
  // Remove all messages (but not the welcome card if it still exists)
  chatWindow.querySelectorAll(".message").forEach((el) => el.remove());
  // Re-insert welcome card if it was removed
  if (!document.getElementById("welcomeCard")) {
    const card = document.createElement("div");
    card.id        = "welcomeCard";
    card.className = "welcome-card";
    card.innerHTML = `
      <div class="welcome-icon">🏥</div>
      <h1 class="welcome-title">Hello, I'm MediBot</h1>
      <p class="welcome-sub">Describe your symptoms and I'll classify them using a <strong>TF-IDF + SVM</strong> machine learning model trained on 1,800 medical examples.</p>
      <div class="quick-examples">
        <p class="quick-label">Try an example:</p>
        <div class="chips" id="quickChips">
          <button class="chip" data-msg="I have itching and rash">Itching &amp; rash</button>
          <button class="chip" data-msg="I have dizziness and fatigue">Dizziness &amp; fatigue</button>
          <button class="chip" data-msg="I have runny nose and sore throat">Runny nose &amp; sore throat</button>
          <button class="chip" data-msg="I have severe headache and nausea">Severe headache &amp; nausea</button>
          <button class="chip" data-msg="I have joint pain and stiffness">Joint pain &amp; stiffness</button>
          <button class="chip" data-msg="I have cough and chest pain">Cough &amp; chest pain</button>
        </div>
      </div>`;
    chatWindow.appendChild(card);
    attachChipListeners();
  }
  // Reset sidebar highlights
  document.querySelectorAll(".intent-chip").forEach((c) =>
    c.classList.remove("active-intent")
  );
}

// ── Quick example chips ───────────────────────────────────────────────────────

function attachChipListeners() {
  document.querySelectorAll(".chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      userInput.value = btn.dataset.msg;
      userInput.dispatchEvent(new Event("input"));
      userInput.focus();
      sendMessage();
    });
  });
}

// ── Textarea auto-resize ──────────────────────────────────────────────────────

userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  const max = 140;
  userInput.style.height = Math.min(userInput.scrollHeight, max) + "px";
  sendBtn.disabled = !userInput.value.trim() || isLoading;
});

// ── Keyboard shortcuts ────────────────────────────────────────────────────────

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ── Event listeners ───────────────────────────────────────────────────────────

sendBtn.addEventListener("click", sendMessage);
clearBtn.addEventListener("click", clearConversation);

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

// Close sidebar on outside click (mobile)
document.addEventListener("click", (e) => {
  if (
    sidebar.classList.contains("open") &&
    !sidebar.contains(e.target) &&
    !sidebarToggle.contains(e.target)
  ) {
    sidebar.classList.remove("open");
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────

sendBtn.disabled = true; // disabled until user types

loadIntents();
attachChipListeners();
userInput.focus();
