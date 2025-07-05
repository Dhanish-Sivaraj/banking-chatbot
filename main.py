import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime, timedelta
import random
from typing import List, Dict
import torch

app = FastAPI()

# Load NLP models
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Initialize conversation model
conversation_model = pipeline(
    "text2text-generation",
    model="facebook/blenderbot-400M-distill",
    tokenizer="facebook/blenderbot-400M-distill",
    device=0 if torch.cuda.is_available() else -1
)

# Initialize banking-specific model
try:
    banking_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    banking_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
except Exception as e:
    print(f"Error loading banking model: {e}")
    # Fallback to simpler model
    banking_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    banking_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Data models
class Query(BaseModel):
    query: str
    user_id: str = "default"
    conversation_history: List[Dict] = []

# Mock database
user_accounts = {
    "user123": {
        "name": "Rahul Sharma",
        "accounts": {
            "savings": {"balance": 187500.50, "number": "XXXXXX7890"},
            "current": {"balance": 325000.75, "number": "XXXXXX1234"}
        },
        "cards": [
            {"type": "credit", "number": "XXXX-4321", "limit": 150000},
            {"type": "debit", "number": "XXXX-8765", "linked": "savings"}
        ],
        "loans": {
            "personal": {"amount": 500000, "emi": 12500, "remaining": 325000},
            "home": {"amount": 8500000, "emi": 62500, "remaining": 7200000}
        },
        "credit_score": 785
    }
}

def get_contextual_response(query: str, conversation_history: List[Dict]) -> str:
    """Generate contextual response using NLP models"""
    try:
        # Try banking-specific model first
        banking_input = "\n".join([msg["content"] for msg in conversation_history[-3:]]) + "\n" + query
        inputs = banking_tokenizer.encode(banking_input + banking_tokenizer.eos_token, return_tensors='pt')
        
        outputs = banking_model.generate(
            inputs,
            max_length=200,
            pad_token_id=banking_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        banking_response = banking_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if any(word in banking_response.lower() for word in ["account", "balance", "loan", "card", "transaction"]):
            return banking_response
        
        # Fall back to general conversation model
        conv_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-3:]])
        return conversation_model(
            f"Conversation history:\n{conv_history}\nUser: {query}\nAssistant:",
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )[0]['generated_text']
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "I'm having trouble understanding that. Could you rephrase your question?"

def process_banking_query(query: str, user_id: str) -> str:
    """Process specific banking queries"""
    user = user_accounts.get(user_id, user_accounts["user123"])
    doc = nlp(query.lower())
    
    # Balance inquiries
    if any(token.text in ["balance", "money", "account"] for token in doc):
        return (
            f"Here are your current balances, {user['name']}:\n\n"
            f"💳 Savings Account ({user['accounts']['savings']['number']}): ₹{user['accounts']['savings']['balance']:,.2f}\n"
            f"🏦 Current Account ({user['accounts']['current']['number']}): ₹{user['accounts']['current']['balance']:,.2f}\n\n"
            f"Last updated: {datetime.now().strftime('%d %b %Y at %H:%M')}"
        )
    
    # Card inquiries
    if any(token.text in ["card", "credit", "debit"] for token in doc):
        response = [f"Here's your card information, {user['name']}:"]
        for card in user['cards']:
            if card['type'] == "credit":
                response.append(
                    f"\n💳 {card['type'].title()} Card (•••• {card['number']})"
                    f"\n  • Limit: ₹{card['limit']:,.2f}"
                )
            else:
                response.append(
                    f"\n💳 {card['type'].title()} Card (•••• {card['number']})"
                    f"\n  • Linked to {card['linked'].title()} Account"
                )
        return "\n".join(response)
    
    # Loan inquiries
    if any(token.text in ["loan", "emi", "borrow"] for token in doc):
        return (
            f"Here's your loan information, {user['name']}:\n\n"
            f"🏠 Home Loan:\n"
            f"  • Amount: ₹{user['loans']['home']['amount']:,.2f}\n"
            f"  • Outstanding: ₹{user['loans']['home']['remaining']:,.2f}\n"
            f"  • EMI: ₹{user['loans']['home']['emi']:,.2f}\n\n"
            f"💼 Personal Loan:\n"
            f"  • Amount: ₹{user['loans']['personal']['amount']:,.2f}\n"
            f"  • Outstanding: ₹{user['loans']['personal']['remaining']:,.2f}\n"
            f"  • EMI: ₹{user['loans']['personal']['emi']:,.2f}\n\n"
            f"Your credit score is {user['credit_score']} (Excellent)."
        )
    
    return None

@app.post("/chat")
async def chat(query: Query):
    try:
        # First try banking-specific responses
        banking_response = process_banking_query(query.query, query.user_id)
        if banking_response:
            return {"response": banking_response}
        
        # Fall back to conversational AI
        contextual_response = get_contextual_response(query.query, query.conversation_history)
        
        # Personalize responses
        user = user_accounts.get(query.user_id, user_accounts["user123"])
        if any(word in query.query.lower() for word in ["thank", "thanks", "appreciate"]):
            contextual_response = random.choice([
                f"You're welcome, {user['name']}! 😊",
                f"Happy to help, {user['name']}!",
                f"My pleasure, {user['name']}! Is there anything else I can assist you with?"
            ])
        elif any(word in query.query.lower() for word in ["hi", "hello", "hey"]):
            contextual_response = random.choice([
                f"Hello {user['name']}! How can I assist you with your banking today?",
                f"Hi there {user['name']}! What banking service can I help you with?",
                f"Good {get_time_of_day()}, {user['name']}! How may I assist you?"
            ])
        
        return {"response": contextual_response}
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return {"response": "Sorry, I'm experiencing technical difficulties. Please try again later."}

def get_time_of_day() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    return "night"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)