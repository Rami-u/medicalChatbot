import json
import random
import torch
import streamlit as st
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F9FC;
}
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #0D1B2A;
    border-right: 1px solid #1E3448;
}
[data-testid="stSidebar"] * { color: #C9D8E8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-family: 'DM Serif Display', serif !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #1A3550 !important;
    color: #7EC8E3 !important;
    border: 1px solid #2A4A66 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 6px 12px !important;
    width: 100% !important;
    text-align: left !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #264D6A !important;
    border-color: #7EC8E3 !important;
    color: #FFFFFF !important;
}

.user-bubble {
    background: #0D6EFD;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px auto;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    width: fit-content;
}
.bot-bubble {
    background: #FFFFFF;
    color: #1E293B;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px auto 6px 0;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.6;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    white-space: pre-wrap;
    width: fit-content;
}
.emergency-bubble {
    background: #FEF2F2;
    color: #991B1B;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px auto 6px 0;
    max-width: 80%;
    font-size: 14px;
    line-height: 1.6;
    border: 1.5px solid #FCA5A5;
    white-space: pre-wrap;
    width: fit-content;
}
.conf-high   { font-size:11px; color:#166534; background:#DCFCE7; padding:2px 8px; border-radius:20px; display:inline-block; margin:4px 0 8px 0; }
.conf-medium { font-size:11px; color:#854D0E; background:#FEF9C3; padding:2px 8px; border-radius:20px; display:inline-block; margin:4px 0 8px 0; }
.conf-low    { font-size:11px; color:#991B1B; background:#FEE2E2; padding:2px 8px; border-radius:20px; display:inline-block; margin:4px 0 8px 0; }

.disclaimer {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 12px;
    color: #9A3412;
    margin-bottom: 16px;
}
.chat-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 20px;
}
.header-icon {
    width: 46px; height: 46px;
    background: linear-gradient(135deg, #0D6EFD, #0DCAF0);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 20px; color: #0D1B2A; margin: 0;
}
.header-sub { font-size: 12px; color: #64748B; margin: 2px 0 0 0; }
.dot { display:inline-block; width:7px; height:7px; background:#22C55E;
       border-radius:50%; margin-right:4px; }
</style>
""", unsafe_allow_html=True)


# ─── Load model and intents ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    data = torch.load("data.pth", map_location=torch.device("cpu"))
    net  = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"])
    net.load_state_dict(data["model_state"])
    net.eval()
    return net, data["all_words"], data["tags"]

@st.cache_resource
def load_intents():
    with open("intents.json", "r") as f:
        return json.load(f)

model, all_words, tags = load_model()
intents_data           = load_intents()


# ─── Session state ────────────────────────────────────────────────────────────
if "messages"        not in st.session_state:
    st.session_state.messages        = []
if "current_context" not in st.session_state:
    st.session_state.current_context = ""
if "quick_msg"       not in st.session_state:
    st.session_state.quick_msg       = None


# ─── Core logic ───────────────────────────────────────────────────────────────
def get_response(user_text):
    tokens = tokenize(user_text)
    bag    = bag_of_words(tokens, all_words)
    bag_t  = torch.tensor(bag, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(bag_t)

    probs           = torch.softmax(output, dim=1)[0]
    confidence, idx = torch.max(probs, dim=0)
    confidence      = confidence.item()
    predicted_tag   = tags[idx.item()]

    best_match = None

    # Pass 1: predicted tag + context check
    for intent in intents_data["intents"]:
        if intent["tag"] == predicted_tag:
            cf = intent.get("context_filter", "")
            if cf == "" or cf == st.session_state.current_context:
                best_match = intent
                break

    # Pass 2: context active — pick best child by word overlap
    if best_match is None and st.session_state.current_context != "":
        stemmed = [stem(t) for t in tokens]
        best_score = -1
        for intent in intents_data["intents"]:
            if intent.get("context_filter", "") == st.session_state.current_context:
                score = sum(
                    1 for p in intent["patterns"]
                    for w in [stem(x) for x in tokenize(p)]
                    if w in stemmed
                )
                if score > best_score:
                    best_score = score
                    best_match = intent

    # Pass 3: fallback unknown
    if best_match is None:
        for intent in intents_data["intents"]:
            if intent["tag"] == "unknown":
                best_match = intent
                break

    st.session_state.current_context = best_match.get("context_set", "") if best_match else ""

    response = random.choice(best_match["responses"]) if best_match else \
               "I could not understand. Please describe your symptoms more clearly."
    tag = best_match["tag"] if best_match else "unknown"
    return response, confidence, tag


def is_emergency(tag):
    return tag in ["emergency", "chest_pain_emergency",
                   "stomach_lower_right", "headache_dangerous"]


def process_message(text):
    """Add user + bot messages to session and rerun."""
    text = text.strip()
    if not text:
        return
    st.session_state.messages.append({"role": "user", "text": text})
    response, confidence, tag = get_response(text)
    st.session_state.messages.append({
        "role": "bot", "text": response,
        "confidence": confidence, "tag": tag
    })


# ─── Sidebar ──────────────────────────────────────────────────────────────────
QUICK = [
    ("🌡️", "I have a fever"),
    ("🤕", "I have a headache"),
    ("😮‍💨", "I have a cough"),
    ("🤢", "My stomach hurts"),
    ("💔", "I have chest pain"),
    ("🔙", "My back hurts"),
    ("😵", "I feel dizzy"),
    ("😴", "I cannot sleep"),
    ("😓", "I feel very tired"),
    ("😰", "I am very stressed"),
]

with st.sidebar:
    st.markdown("## 🩺 Health Assistant")
    st.markdown("---")
    st.markdown("### Quick Questions")
    st.caption("Tap any symptom below")

    for emoji, question in QUICK:
        if st.button(f"{emoji}  {question}", key=f"q_{question}"):
            st.session_state.quick_msg = question

    st.markdown("---")
    if st.button("🗑️  Clear conversation"):
        st.session_state.messages        = []
        st.session_state.current_context = ""
        st.session_state.quick_msg       = None
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.caption("Neural Network trained on 40 medical intents · 498 symptom patterns · Context-aware follow-up system.")
    st.markdown("⚠️ **Not a real doctor.** Always see a physician for serious symptoms.")


# ─── Handle quick question BEFORE rendering ───────────────────────────────────
# This must run before display so the message appears immediately
if st.session_state.quick_msg:
    msg = st.session_state.quick_msg
    st.session_state.quick_msg = None
    process_message(msg)
    st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <div class="header-icon">🩺</div>
    <div>
        <p class="header-title">Health Assistant</p>
        <p class="header-sub"><span class="dot"></span>Online · AI symptom checker</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <b>Disclaimer:</b> This is an AI assistant — NOT a real doctor.
    For serious or persistent symptoms, always consult a licensed physician.
    In a life-threatening emergency, call emergency services immediately.
</div>
""", unsafe_allow_html=True)


# ─── Display messages ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#94A3B8;">
        <div style="font-size:50px; margin-bottom:12px;">💬</div>
        <div style="font-family:'DM Serif Display',serif; font-size:20px;
                    color:#475569; margin-bottom:8px;">How are you feeling today?</div>
        <div style="font-size:14px;">Type your symptom below or tap a quick question from the sidebar.</div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    role = msg["role"]
    text = msg["text"]

    if role == "user":
        st.markdown(f'<div class="user-bubble">{text}</div>',
                    unsafe_allow_html=True)
    else:
        tag  = msg.get("tag", "")
        conf = msg.get("confidence", None)

        bubble = "emergency-bubble" if is_emergency(tag) else "bot-bubble"
        icon   = "🚨" if is_emergency(tag) else "🩺"

        st.markdown(f'<div class="{bubble}">{icon} {text}</div>',
                    unsafe_allow_html=True)

        if conf is not None:
            pct = int(conf * 100)
            if pct >= 80:
                st.markdown(f'<span class="conf-high">✓ {pct}% confident</span>',
                            unsafe_allow_html=True)
            elif pct >= 60:
                st.markdown(f'<span class="conf-medium">~ {pct}% confident</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="conf-low">? {pct}% confident</span>',
                            unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)


# ─── Chat input ───────────────────────────────────────────────────────────────
# st.chat_input() is Streamlit's native chat input widget.
# It correctly handles Enter key and rerun — no conflicts with session state.
user_input = st.chat_input("Describe your symptoms here...")

if user_input:
    process_message(user_input)
    st.rerun()