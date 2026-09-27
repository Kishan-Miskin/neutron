import os
import tempfile

# Redirect all HuggingFace hub downloads and temporary file staging to D drive (123GB free space)
# preventing 'IO Error: There is not enough space on the disk' on C drive (~800MB free)
CACHE_DIR = "d:/AI/.cache"
TMP_DIR = "d:/AI/.cache/tmp"
os.makedirs(TMP_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_CACHE"] = CACHE_DIR
os.environ["TMPDIR"] = TMP_DIR
os.environ["TEMP"] = TMP_DIR
os.environ["TMP"] = TMP_DIR
tempfile.tempdir = TMP_DIR

import streamlit as st
import torch
import tiktoken
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpt2_scratch import GPT2

# =============================================================================
# 1. Page Configuration & Notion-Style Custom CSS
# =============================================================================
st.set_page_config(
    page_title="Notion AI",
    page_icon="https://www.notion.so/images/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS: Pitch Black Theme, Notion Typography, Vectors, Clean Alignment
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        letter-spacing: -0.01em;
    }

    /* PURE PITCH-BLACK BACKGROUND */
    .stApp {
        background-color: #0f0f0f !important;
        color: #efefef;
    }

    /* SIDEBAR (NOTION DARK CANVAS) */
    section[data-testid="stSidebar"] {
        background-color: #141414 !important;
        border-right: 1px solid #222222 !important;
        width: 260px !important;
    }

    /* Center Main Canvas Responsively */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 7rem !important;
        max-width: 780px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Sidebar Logo Header */
    .notion-sidebar-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 4px 16px 4px;
        border-bottom: 1px solid #222222;
        margin-bottom: 16px;
    }

    .notion-logo-text {
        font-size: 0.95rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.02em;
    }

    /* Model Selectbox Custom Styling */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #181818 !important;
        border: 1px solid #282828 !important;
        color: #e5e5e5 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
    }

    /* New Chat Button */
    div.stButton > button {
        background-color: #1c1c1c !important;
        border: 1px solid #2d2d2d !important;
        color: #e3e3e3 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 8px !important;
        transition: background 0.15s ease !important;
    }

    div.stButton > button:hover {
        background-color: #262626 !important;
        border-color: #383838 !important;
    }

    /* Sidebar Section Titles */
    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #737373;
        margin: 18px 0 8px 4px;
    }

    /* Profile Footer Card */
    .notion-profile-card {
        position: fixed;
        bottom: 16px;
        left: 14px;
        width: 232px;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 6px;
        background-color: #181818;
        border: 1px solid #262626;
        z-index: 10;
    }

    .notion-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #2b2b2b;
        color: #e5e5e5;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #3d3d3d;
    }

    .notion-profile-name {
        font-size: 0.85rem;
        font-weight: 500;
        color: #f5f5f5;
        line-height: 1.2;
    }

    .notion-profile-plan {
        font-size: 0.72rem;
        color: #737373;
    }

    /* Main Area Hero Header */
    .hero-container {
        text-align: center;
        margin-top: 18vh;
        margin-bottom: 2rem;
    }

    .hero-title-text {
        font-size: 2.2rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }

    .hero-sub-text {
        font-size: 0.95rem;
        color: #737373;
        font-weight: 400;
    }

    /* Messages Formatting */
    .msg-row-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }

    .msg-user-bubble {
        background-color: #232323;
        border: 1px solid #333333;
        color: #f5f5f5;
        padding: 12px 18px;
        border-radius: 18px;
        max-width: 75%;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .msg-row-assistant {
        display: flex;
        gap: 14px;
        align-items: flex-start;
        margin-bottom: 28px;
        max-width: 90%;
    }

    .ai-icon-badge {
        width: 28px;
        height: 28px;
        border-radius: 6px;
        background-color: #1c1c1c;
        border: 1px solid #333333;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }

    .assistant-body {
        color: #e5e5e5;
        font-size: 0.98rem;
        line-height: 1.65;
        font-weight: 400;
    }

    /* Streamlit Native Bottom Fixed Input Styling - Responsively Centered */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-baseweb="input"] {
        position: fixed !important;
        bottom: 24px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: min(760px, 85vw) !important;
        background-color: #181818 !important;
        border: 1px solid #333333 !important;
        border-radius: 24px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
        z-index: 99999 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    /* PURE WHITE HOVER & FOCUS BORDER EFFECT (NO RED) */
    div[data-testid="stChatInput"]:hover,
    div[data-testid="stChatInput"]:hover > div,
    div[data-testid="stChatInput"]:hover [data-baseweb="base-input"],
    div[data-testid="stChatInput"]:hover [data-baseweb="input"],
    div[data-testid="stChatInput"]:focus-within,
    div[data-testid="stChatInput"]:focus-within > div,
    div[data-testid="stChatInput"]:focus-within [data-baseweb="base-input"],
    div[data-testid="stChatInput"]:focus-within [data-baseweb="input"] {
        border-color: #ffffff !important;
        box-shadow: 0 0 16px rgba(255, 255, 255, 0.25) !important;
        outline: none !important;
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] textarea:focus {
        color: #ffffff !important;
        background-color: transparent !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. Model Loaders
# =============================================================================
@st.cache_resource
def load_qwen_engine():
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    fallback_name = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback_name, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(fallback_name, cache_dir=CACHE_DIR)
    return tokenizer, model

@st.cache_resource
def load_gpt2_engine():
    model = GPT2.from_pretrained("gpt2")
    if os.path.exists("gpt2_slimorca_trained.pt"):
        model.load_state_dict(torch.load("gpt2_slimorca_trained.pt", weights_only=True))
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer, model

# =============================================================================
# 3. Session State & Chat History Management
# =============================================================================
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = "Chat 1"
    st.session_state.conversations["Chat 1"] = []

messages = st.session_state.conversations[st.session_state.active_chat_id]

# Vector SVG Icon
SVG_SPARKLES = '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#a3a3a3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z"></path></svg>'''

# =============================================================================
# 4. Sidebar Layout
# =============================================================================
with st.sidebar:
    st.markdown(f'''
    <div class="notion-sidebar-header">
        {SVG_SPARKLES}
        <span class="notion-logo-text">Notion AI</span>
    </div>
    ''', unsafe_allow_html=True)
    
    # Model Selection Switcher
    st.markdown('<div class="sidebar-section-title">Model Engine</div>', unsafe_allow_html=True)
    selected_model_option = st.selectbox(
        "Select Model",
        options=["Smart AI Engine (Instant / Accurate)", "Custom GPT-2 (124M SlimOrca)"],
        label_visibility="collapsed"
    )

    # New Chat Button
    if st.button("New chat", key="btn_new_chat_notion", use_container_width=True):
        new_id = f"Chat {len(st.session_state.conversations) + 1}"
        st.session_state.conversations[new_id] = []
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.markdown('<div class="sidebar-section-title">Recent Chats</div>', unsafe_allow_html=True)

    # Render History List
    chat_ids = list(st.session_state.conversations.keys())
    for cid in reversed(chat_ids):
        chat_msgs = st.session_state.conversations[cid]
        if len(chat_msgs) > 0 and chat_msgs[0]["role"] == "user":
            title = chat_msgs[0]["content"]
            if len(title) > 22:
                title = title[:22] + "..."
        else:
            title = cid
        
        is_active = (cid == st.session_state.active_chat_id)
        prefix = "• " if is_active else "  "
        
        if st.button(f"{prefix}{title}", key=f"hist_{cid}", use_container_width=True):
            st.session_state.active_chat_id = cid
            st.rerun()

    # User Profile Card at Sidebar Bottom
    st.markdown("""
    <div class="notion-profile-card">
        <div class="notion-avatar">KM</div>
        <div>
            <div class="notion-profile-name">Kishan Miskin</div>
            <div class="notion-profile-plan">Free Workspace</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 5. Main Canvas & Conversation Trajectory
# =============================================================================
if len(messages) == 0:
    st.markdown('''
    <div class="hero-container">
        <div class="hero-title-text">What can I help with today?</div>
        <div class="hero-sub-text">Ask anything or type a prompt to begin</div>
    </div>
    ''', unsafe_allow_html=True)
else:
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f'''
            <div class="msg-row-user">
                <div class="msg-user-bubble">{msg["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="msg-row-assistant">
                <div class="ai-icon-badge">{SVG_SPARKLES}</div>
                <div class="assistant-body">{msg["content"]}</div>
            </div>
            ''', unsafe_allow_html=True)

# Spacing before fixed bottom input bar
st.markdown('<div style="height: 100px;"></div>', unsafe_allow_html=True)

# =============================================================================
# 6. Fixed Bottom Chat Input Bar & Generation
# =============================================================================
user_input = st.chat_input("Ask anything...")

if user_input:
    # 1. Append user message
    messages.append({"role": "user", "content": user_input.strip()})
    query = user_input.strip()
    
    with st.spinner(""):
        if "Smart" in selected_model_option:
            # Smart AI Engine (Instant & Accurate)
            qwen_tok, qwen_mod = load_qwen_engine()
            
            # Format context history into ChatML
            chat_history = []
            for m in messages:
                chat_history.append({"role": m["role"], "content": m["content"]})
                
            prompt_text = qwen_tok.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            inputs = qwen_tok(prompt_text, return_tensors="pt")
            
            with torch.no_grad():
                gen_tokens = qwen_mod.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=qwen_tok.eos_token_id
                )
            
            ans = qwen_tok.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        else:
            # Custom GPT-2 Scratch Engine (124M SlimOrca)
            gpt2_enc, gpt2_mod = load_gpt2_engine()
            formatted_prompt = f"Q: {query}\nA:"
            
            tokens = torch.tensor([gpt2_enc.encode(formatted_prompt, allowed_special="all")], dtype=torch.long)
            with torch.no_grad():
                gen_tokens = gpt2_mod.generate(
                    tokens,
                    max_new_tokens=150,
                    temperature=0.3,
                    top_k=25
                )
            raw_output = gpt2_enc.decode(gen_tokens[0].tolist())
            
            if "A:" in raw_output:
                ans = raw_output.split("A:", 1)[1].strip()
                if "\nQ:" in ans:
                    ans = ans.split("\nQ:", 1)[0].strip()
            else:
                ans = raw_output
            ans = ans.replace("<|endoftext|>", "").strip()

    # 2. Append assistant message & refresh UI
    messages.append({"role": "assistant", "content": ans})
    st.rerun()
