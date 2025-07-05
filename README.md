# 💬 NeoBank AI Assistant

A Conversational AI Banking Assistant powered by Streamlit (Frontend) and FastAPI (Backend), capable of handling banking queries like:

- ✅ Account balance
- 💳 Credit/Debit card details
- 🏦 Loan info
- 📤 Fund transfer guidance
- 🤖 Casual small talk

## 🛠️ Features

- Dual model handling (DistilGPT2, DialoGPT, BlenderBot)
- Streamed assistant response with typing animation
- Sidebar with Quick Actions
- Custom HTML-styled banking responses
- Contextual chat memory
- Modular FastAPI backend for scalability

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/neobank-ai-assistant.git
cd neobank-ai-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Optional: For better model performance, also run:  
`python -m spacy download en_core_web_md`

## 🚀 Run the Project

### 🔹 Start the Backend (FastAPI)
```bash
uvicorn main:app --reload
```

> Access it at: http://127.0.0.1:8000

### 🔹 Start the Frontend (Streamlit)
```bash
streamlit run app.py
```

## 📁 File Structure

```
neobank-ai-assistant/
├── app.py            # Streamlit frontend
├── main.py           # FastAPI backend
├── requirements.txt  # Python dependencies
├── README.md         # Project overview
└── .gitignore
```

## 📬 Contact

Built by **Your Name**  
📧 Email: your.email@example.com  
🌐 [LinkedIn](https://www.linkedin.com/in/your-profile)
