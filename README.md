<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=200&section=header&text=Next%20Word%20Predictor&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=LSTM%20Neural%20Network%20%7C%20TensorFlow%20%7C%20Streamlit&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<br/>

[![Live Demo](https://img.shields.io/badge/🔮%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logoColor=white)](https://ashabakshi-nextwordpre-lstm13.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-ashabakshi-181717?style=for-the-badge&logo=github)](https://github.com/ashabakshi/ashabakshi-NextWordPre-LSTM)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-LSTM-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)

<br/>

> *"Type a sentence. Let AI complete your thought."*

</div>

---

## 🧠 What is This?

A deep learning app that **predicts the next word(s)** in a sentence using an **LSTM (Long Short-Term Memory)** neural network — trained on a curated **Quotes Dataset**.

Type any phrase, hit **Predict**, and watch the model suggest what comes next — complete with confidence scores and a visual probability breakdown.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Next Word Prediction** | Predicts 1–10 words sequentially |
| 📊 **Top 5 Candidates** | Shows top predictions with confidence % bar |
| 📝 **Sentence Builder** | Visual chip view of original + predicted words |
| 📈 **Live Stats** | Input words, predicted count, vocab size, seq length |
| 🎨 **Animated Dark UI** | Glassmorphism design with gradient animations |
| ⚡ **Cached Model Loading** | Fast inference with `@st.cache_resource` |

---

## 🚀 Try It Live

**👉 [ashabakshi-nextwordpre-lstm13.streamlit.app](https://ashabakshi-nextwordpre-lstm13.streamlit.app/)**



<div align="center">
<img src="https://img.shields.io/badge/Status-Live%20%26%20Deployed-22c55e?style=flat-square&logo=streamlit" />
</div>

---

## 🏗️ How It Works

```
User Input (text)
       ↓
Tokenizer → Token Sequences
       ↓
Padding (pre-pad to max_len - 1)
       ↓
LSTM Model → Softmax Output (vocab_size probabilities)
       ↓
Top-N Predictions + Confidence Scores
       ↓
Sequential prediction for multi-word output
```

---

## 🗂️ Project Structure

```
Next_word_predictor/
│
├── app.py               # Streamlit app — UI + prediction logic
├── lstm_model.h5        # Trained LSTM model (Keras)
├── tokenizer.pkl        # Fitted tokenizer (word → index mapping)
├── max_len.pkl          # Max sequence length used during training
├── requirements.txt     # Python dependencies
├── runtime.txt          # Python version for Streamlit Cloud
└── .python-version      # Local Python version pin
```

---

## 🧩 Model Architecture

| Component | Detail |
|---|---|
| **Model Type** | LSTM (Long Short-Term Memory) |
| **Framework** | TensorFlow / Keras |
| **Training Data** | Quotes Dataset |
| **Task** | Multi-class next-word classification |
| **Output Layer** | Softmax over full vocabulary |
| **Prediction** | `np.argsort` → Top-N words + normalized confidence |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/ashabakshi/ashabakshi-NextWordPre-LSTM.git
cd ashabakshi-NextWordPre-LSTM

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

> Make sure `lstm_model.h5`, `tokenizer.pkl`, and `max_len.pkl` are in the same directory as `app.py`.

---

## 📦 Dependencies

```txt
streamlit
tensorflow==2.15.0
keras==2.15.0
numpy
```

Full list in [`requirements.txt`](./requirements.txt)

---

## 🎯 Example Predictions

| Input | Predicted Word(s) |
|---|---|
| `The future of artificial intelligence is` | `the` / `a` / `not` |
| `In the end, what matters most is` | `the` / `your` / `love` |
| `Success is not final, failure is not` | `a` / `the` / `always` |

---

## 👩‍💻 About the Developer

<div align="center">

**Asha Bakshi**
BCA Final Year · Aspiring Data Scientist

[![GitHub](https://img.shields.io/badge/GitHub-ashabakshi-181717?style=flat-square&logo=github)](https://github.com/ashabakshi)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=100&section=footer&animation=fadeIn" width="100%"/>

*Built with ❤️ using LSTM + TensorFlow + Streamlit*

</div>
