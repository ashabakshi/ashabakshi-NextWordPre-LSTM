import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Next Word Predictor | LSTM AI",
    page_icon="🔮",
    layout="centered"
)

# ---------------- GOOGLE FONTS ----------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* ===== GLOBAL ===== */
.stApp {
    background: #0a0a0f;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}

/* Animated gradient mesh background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background:
        radial-gradient(ellipse 600px 600px at 10% 20%, rgba(124,58,237,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 500px 500px at 80% 10%, rgba(236,72,153,0.10) 0%, transparent 70%),
        radial-gradient(ellipse 400px 400px at 50% 80%, rgba(59,130,246,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 300px 300px at 90% 70%, rgba(16,185,129,0.06) 0%, transparent 70%);
    animation: meshMove 15s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
}

@keyframes meshMove {
    0% { transform: scale(1) rotate(0deg); }
    50% { transform: scale(1.05) rotate(1deg); }
    100% { transform: scale(1) rotate(-1deg); }
}

/* ===== HIDE STREAMLIT DEFAULTS ===== */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 750px; }

/* ===== HEADER ===== */
.hero-container {
    text-align: center;
    padding: 2rem 0 1.5rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(236,72,153,0.15));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 50px;
    padding: 6px 18px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin: 0.5rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #a78bfa 40%, #ec4899 70%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleShimmer 4s ease-in-out infinite alternate;
}

@keyframes titleShimmer {
    0% { filter: brightness(1); }
    50% { filter: brightness(1.2); }
    100% { filter: brightness(1); }
}

.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 400;
    margin-top: 0.5rem;
    line-height: 1.6;
}

.hero-subtitle span {
    color: #a78bfa;
    font-weight: 500;
}

/* ===== DIVIDER ===== */
.gradient-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.4), rgba(236,72,153,0.4), transparent);
    margin: 1.5rem 0 2rem;
    border: none;
}

/* ===== GLASS CARD ===== */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124,58,237,0.5), rgba(236,72,153,0.5), transparent);
}

.card-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1rem;
}

/* ===== INPUT STYLING ===== */
.stTextInput > div > div > input {
    background: rgba(15,15,25,0.8) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(124,58,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1), 0 0 30px rgba(124,58,237,0.1) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #475569 !important;
}

.stTextInput label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ===== NUMBER INPUT ===== */
.stNumberInput > div > div > input {
    background: rgba(15,15,25,0.8) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
    border-radius: 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stNumberInput label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    border: none !important;
    padding: 14px 24px !important;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.45) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ===== RESULT CARD ===== */
.result-container {
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(236,72,153,0.05));
    border: 1px solid rgba(124,58,237,0.15);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
    animation: resultSlideIn 0.5s cubic-bezier(0.4,0,0.2,1);
}

.result-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #7c3aed, #ec4899, #7c3aed);
    background-size: 200% 100%;
    animation: borderGlow 3s linear infinite;
}

@keyframes borderGlow {
    0% { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

@keyframes resultSlideIn {
    from {
        opacity: 0;
        transform: translateY(20px) scale(0.98);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.8rem;
}

.result-word {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.5px;
}

.result-sentence {
    margin-top: 1.2rem;
    padding: 1rem 1.2rem;
    background: rgba(0,0,0,0.3);
    border-radius: 12px;
    border-left: 3px solid #7c3aed;
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.7;
    font-family: 'Inter', sans-serif;
}

.result-sentence .highlight {
    color: #a78bfa;
    font-weight: 700;
    text-decoration: underline;
    text-decoration-color: rgba(167,139,250,0.3);
    text-underline-offset: 3px;
}

/* ===== TOP PREDICTIONS ===== */
.predictions-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 1.2rem;
}

.pred-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 12px;
    transition: all 0.2s ease;
    animation: predFadeIn 0.4s ease forwards;
    opacity: 0;
}

.pred-item:hover {
    background: rgba(124,58,237,0.06);
    border-color: rgba(124,58,237,0.15);
    transform: translateX(4px);
}

@keyframes predFadeIn {
    to { opacity: 1; }
}

.pred-rank {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}

.rank-1 { background: linear-gradient(135deg, #7c3aed, #a78bfa); color: white; }
.rank-2 { background: rgba(124,58,237,0.2); color: #a78bfa; }
.rank-3 { background: rgba(124,58,237,0.12); color: #8b5cf6; }
.rank-4 { background: rgba(124,58,237,0.08); color: #7c3aed; }
.rank-5 { background: rgba(124,58,237,0.05); color: #6d28d9; }

.pred-word {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 600;
    color: #e2e8f0;
    min-width: 100px;
}

.pred-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    overflow: hidden;
}

.pred-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #7c3aed, #ec4899);
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
}

.pred-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    min-width: 50px;
    text-align: right;
}

/* ===== SENTENCE BUILDER ===== */
.builder-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(124,58,237,0.1);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-top: 0.8rem;
    min-height: 60px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
}

.word-chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
}

.word-original { background: rgba(100,116,139,0.15); color: #94a3b8; }
.word-predicted {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(236,72,153,0.15));
    color: #c4b5fd;
    border: 1px solid rgba(124,58,237,0.2);
}

/* ===== STATS ===== */
.stats-row {
    display: flex;
    gap: 12px;
    margin-top: 1.5rem;
}

.stat-chip {
    flex: 1;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
}

.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #a78bfa;
}

.stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #475569;
    margin-top: 4px;
}

/* ===== WARNING ===== */
.stAlert {
    border-radius: 14px !important;
}

/* ===== FLOATING ORBS (decorative) ===== */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
    z-index: -1;
}

.orb-1 {
    width: 300px; height: 300px;
    background: rgba(124,58,237,0.08);
    top: 10%; left: -5%;
    animation: orbFloat1 20s ease-in-out infinite;
}

.orb-2 {
    width: 250px; height: 250px;
    background: rgba(236,72,153,0.06);
    bottom: 10%; right: -5%;
    animation: orbFloat2 25s ease-in-out infinite;
}

@keyframes orbFloat1 {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(30px, 50px); }
}

@keyframes orbFloat2 {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(-40px, -30px); }
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 2rem 0 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    color: #475569;
    font-size: 0.8rem;
    letter-spacing: 0.3px;
}

.footer-name {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.6rem;
}

.footer-name span {
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.footer-links {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    margin: 0.8rem 0;
}

.footer-links a {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: #64748b;
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
    transition: all 0.25s ease;
}

.footer-links a:hover {
    color: #a78bfa;
    border-color: rgba(124,58,237,0.25);
    background: rgba(124,58,237,0.06);
    transform: translateY(-1px);
}

.footer-links a svg {
    width: 16px;
    height: 16px;
    fill: currentColor;
}

.footer-info {
    font-size: 0.72rem;
    color: #334155;
    margin-top: 0.6rem;
    letter-spacing: 0.5px;
}

.footer-info span {
    color: #7c3aed;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Floating orbs (decorative)
st.markdown("""
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES (cached) ----------------
@st.cache_resource
def load_assets():
    model = load_model("lstm_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return model, tokenizer, max_len

model, tokenizer, max_len = load_assets()

vocab_size = len(tokenizer.word_index) + 1

# Fast reverse-lookup dictionary (index → word)
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🧠 LSTM Neural Network</div>
    <div class="hero-title">Next Word Predictor</div>
    <div class="hero-subtitle">
        Type a sentence and let the AI predict what comes next.<br>
        Powered by <span>Long Short-Term Memory</span> · Trained on a <span>Quotes Dataset</span>
    </div>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "built_sentence" not in st.session_state:
    st.session_state.built_sentence = ""
if "predicted_words" not in st.session_state:
    st.session_state.predicted_words = []

# ---------------- PREDICTION FUNCTION ----------------
def predict_next_words(text, top_n=5):
    """Predict top N next words with confidence scores."""
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]

        # Handle empty / unrecognized input
        if len(token_list) == 0:
            return []

        token_list = pad_sequences(
            [token_list],
            maxlen=max_len - 1,
            padding='pre'
        )
        predictions = model.predict(token_list, verbose=0)[0]

        # Get top N predictions
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        top_probs = predictions[top_indices]

        # Normalize to percentages
        total = np.sum(top_probs)
        if total > 0:
            top_probs = top_probs / total * 100

        results = []
        for idx, prob in zip(top_indices, top_probs):
            word = index_to_word.get(idx, "")
            if word:
                results.append((word, float(prob)))

        return results
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return []

# ---------------- INPUT SECTION ----------------
st.markdown("""
<div class="glass-card">
    <div class="card-label">⌨️ Input</div>
</div>
""", unsafe_allow_html=True)

# Slightly overlap with the card above using negative margin via columns
input_text = st.text_input(
    "Enter your sentence",
    placeholder="The future of artificial intelligence is...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 1])
with col1:
    predict_btn = st.button("🔮  Predict Next Word", use_container_width=True)
with col2:
    num_words = st.number_input(
        "Words",
        min_value=1,
        max_value=10,
        value=1,
        label_visibility="collapsed",
        help="Number of words to predict sequentially"
    )

# ---------------- PREDICTION LOGIC ----------------
if predict_btn:
    if not input_text or input_text.strip() == "":
        st.warning("⚠️ Please enter some text to get a prediction.")
    else:
        current_text = input_text.strip()
        st.session_state.built_sentence = current_text
        st.session_state.predicted_words = []

        # Multi-word sequential prediction
        for i in range(int(num_words)):
            results = predict_next_words(current_text)
            if results:
                best_word = results[0][0]
                st.session_state.predicted_words.append(best_word)
                current_text += " " + best_word

        # ===== RESULT DISPLAY =====
        full_predicted = " ".join(st.session_state.predicted_words)

        st.markdown(f"""
        <div class="result-container">
            <div class="result-label">✨ Predicted {"Words" if num_words > 1 else "Word"}</div>
            <div class="result-word">{full_predicted}</div>
            <div class="result-sentence">
                {input_text.strip()} <span class="highlight">{full_predicted}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ===== SENTENCE BUILDER =====
        st.markdown("""
        <div class="glass-card" style="margin-top: 1.5rem;">
            <div class="card-label">📝 Sentence Builder</div>
        """, unsafe_allow_html=True)

        original_words = input_text.strip().split()
        chips_html = ""
        for w in original_words:
            chips_html += f'<span class="word-chip word-original">{w}</span> '
        for w in st.session_state.predicted_words:
            chips_html += f'<span class="word-chip word-predicted">{w}</span> '

        st.markdown(f"""
            <div class="builder-box">{chips_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ===== TOP PREDICTIONS (for last word) =====
        final_results = predict_next_words(
            input_text.strip() if num_words == 1 else current_text.rsplit(" ", 1)[0],
            top_n=5
        )

        if final_results:
            st.markdown("""
            <div class="glass-card" style="margin-top: 1.5rem;">
                <div class="card-label">📊 Top Predictions</div>
                <div class="predictions-grid">
            """, unsafe_allow_html=True)

            for i, (word, conf) in enumerate(final_results):
                delay = i * 0.1
                st.markdown(f"""
                <div class="pred-item" style="animation-delay: {delay}s;">
                    <div class="pred-rank rank-{i+1}">{i+1}</div>
                    <div class="pred-word">{word}</div>
                    <div class="pred-bar-bg">
                        <div class="pred-bar-fill" style="width: {conf:.1f}%;"></div>
                    </div>
                    <div class="pred-pct">{conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        # ===== STATS =====
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-chip">
                <div class="stat-value">{len(input_text.strip().split())}</div>
                <div class="stat-label">Input Words</div>
            </div>
            <div class="stat-chip">
                <div class="stat-value">{int(num_words)}</div>
                <div class="stat-label">Predicted</div>
            </div>
            <div class="stat-chip">
                <div class="stat-value">{vocab_size:,}</div>
                <div class="stat-label">Vocabulary</div>
            </div>
            <div class="stat-chip">
                <div class="stat-value">{max_len}</div>
                <div class="stat-label">Seq Length</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <div class="footer-name">Made by <span>Asha Bakshi</span></div>
    <div class="footer-links">
        <a href="https://github.com/ashabakshi/NextWord-LSTM" target="_blank">
            <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
            GitHub
        </a>
    </div>
    <div class="footer-info">
        Built with <span>LSTM</span> + TensorFlow · Powered by <span>Streamlit</span> · Trained on <span>Quotes Dataset</span>
    </div>
</div>
""", unsafe_allow_html=True)