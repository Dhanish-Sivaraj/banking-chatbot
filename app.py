import streamlit as st
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import random
import torch
from typing import List, Dict
import time
import sys
from functools import lru_cache

# Initial loading screen
if 'ready' not in st.session_state:
    st.title("NeoBank AI Assistant")
    with st.status("Initializing banking assistant...", expanded=True) as status:
        st.write("Loading core modules...")
        time.sleep(0.5)
        st.write("Setting up security...")
        time.sleep(0.3)
        st.write("Almost ready...")
        time.sleep(0.2)
        status.update(label="Ready!", state="complete")
    st.session_state.ready = True
    st.rerun()

# Custom CSS for chat interface
st.markdown("""
<style>
    .stChatInput {
        position: fixed;
        bottom: 3rem;
        width: 80%;
        left: 10%;
        border-radius: 25px;
        padding: 12px 20px;
    }
    .stChatMessage {
        padding: 1.2rem;
        border-radius: 18px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        line-height: 1.6;
    }
    [data-testid="stChatMessage"][aria-label="Chat message from user"] {
        margin-left: 20%;
        background-color: #0078D4;
        color: white;
    }
    [data-testid="stChatMessage"][aria-label="Chat message from assistant"] {
        margin-right: 20%;
        background-color: #F5F7FA;
        border: 1px solid #E5E7EB;
    }
    .banking-response {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .banking-response h2 {
        color: #2e86de;
        margin-bottom: 0.5rem;
    }
    .banking-response h3 {
        color: #54a0ff;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .banking-response ul {
        padding-left: 1.2rem;
    }
    .typing-indicator {
        display: inline-block;
        padding: 0 8px;
    }
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #888;
        margin: 0 2px;
        animation: typingAnimation 1.4s infinite ease-in-out;
    }
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typingAnimation {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize models with better error handling
@st.cache_resource(ttl=3600, show_spinner="Loading AI models...")
def load_models():
    models = {}
    try:
        # Load spaCy NLP model
        try:
            models['nlp'] = spacy.load("en_core_web_sm")
        except:
            import en_core_web_sm
            models['nlp'] = en_core_web_sm.load()
        
        # Load lighter conversation model
        models['conversation_model'] = pipeline(
            "text-generation",
            model="distilgpt2",
            device="cpu"
        )
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None

# Lazy load banking model
def load_banking_model():
    if 'banking_model' not in st.session_state:
        with st.spinner("Loading banking features..."):
            try:
                st.session_state.banking_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
                st.session_state.banking_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
                return True
            except Exception as e:
                st.error(f"Couldn't load banking model: {str(e)}")
                return False
    return True

# Stream response with typing effect
def stream_response(response):
    message_placeholder = st.empty()
    full_response = ""
    
    # Split by sentences for better performance
    for chunk in response.split(". "):
        full_response += chunk + ". "
        time.sleep(0.08)
        message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)

# Enhanced mock user database
user_accounts = {
    "user123": {
        "name": "User",
        "accounts": {
            "savings": {"balance": 187500.50, "number": "XXXXXX7890", "transactions": [
                {"date": "2023-06-15", "description": "Salary Credit", "amount": 75000.00},
                {"date": "2023-06-10", "description": "Utility Payment", "amount": -4500.00}
            ]},
            "current": {"balance": 325000.75, "number": "XXXXXX1234", "transactions": [
                {"date": "2023-06-14", "description": "Client Payment", "amount": 125000.00},
                {"date": "2023-06-12", "description": "Vendor Payment", "amount": -35000.00}
            ]}
        },
        "cards": [
            {"type": "credit", "number": "XXXX-4321", "limit": 150000, "due_date": "25th", "outstanding": 45000.00},
            {"type": "debit", "number": "XXXX-8765", "linked": "savings", "daily_limit": 50000.00}
        ],
        "loans": {
            "personal": {"amount": 500000, "emi": 12500, "remaining": 325000, "interest_rate": "12.5%"},
            "home": {"amount": 8500000, "emi": 62500, "remaining": 7200000, "interest_rate": "8.75%"}
        },
        "credit_score": 785,
        "preferences": {"language": "English", "notification": True}
    }
}

def get_banking_response(query: str, user_id: str = "user123") -> str:
    """Enhanced banking query handler with more features"""
    user = user_accounts.get(user_id, user_accounts["user123"])
    doc = nlp(query.lower())
    
    # Balance inquiries
    if any(token.text in ["balance", "money", "account"] for token in doc):
        return f"""
<div class='banking-response'>
<h2>Account Balances</h2>
<p>Here are your current balances, {user['name']}:</p>
<ul>
<li>💰 <strong>Savings Account</strong> (••••7890): ₹{user['accounts']['savings']['balance']:,.2f}</li>
<li>💳 <strong>Current Account</strong> (••••1234): ₹{user['accounts']['current']['balance']:,.2f}</li>
</ul>
<p><em>Last updated: {datetime.now().strftime('%d %b %Y at %H:%M')}</em></p>

<h3>How else may I assist you?</h3>
<ul>
<li>View recent transactions</li>
<li>Transfer funds</li>
<li>Request statement</li>
<li>Update account details</li>
</ul>
</div>
"""
    
    # Transaction inquiries
    if any(token.text in ["transaction", "history", "statement"] for token in doc):
        response = [f"""
<div class='banking-response'>
<h2>Recent Transactions</h2>
<p>Here are your recent transactions, {user['name']}:</p>
"""]
        
        for account_type, account in user['accounts'].items():
            response.append(f"<h3>{account_type.title()} Account (••••{account['number'][-4:]})</h3>")
            response.append("<ul>")
            for tx in account['transactions'][-5:]:
                amount = f"+₹{tx['amount']:,.2f}" if tx['amount'] > 0 else f"-₹{abs(tx['amount']):,.2f}"
                response.append(f"<li>{tx['date']}: {tx['description']} - {amount}</li>")
            response.append("</ul>")
        
        response.append("""
<h3>Transaction Services</h3>
<ul>
<li>Download full statement</li>
<li>Dispute a transaction</li>
<li>Set up alerts</li>
</ul>
</div>
""")
        return "\n".join(response)
    
    # Card inquiries
    if any(token.text in ["card", "credit", "debit"] for token in doc):
        cards_info = []
        for card in user['cards']:
            if card['type'] == "credit":
                cards_info.append(f"""
<li>💳 <strong>{card['type'].title()} Card</strong> (••••{card['number'][-4:]})
<ul>
<li>Credit Limit: ₹{card['limit']:,.2f}</li>
<li>Outstanding: ₹{card['outstanding']:,.2f}</li>
<li>Payment Due: {card['due_date']} of each month</li>
</ul>
</li>
""")
            else:
                cards_info.append(f"""
<li>💳 <strong>{card['type'].title()} Card</strong> (••••{card['number'][-4:]})
<ul>
<li>Linked to: {card['linked'].title()} Account</li>
<li>Daily Limit: ₹{card['daily_limit']:,.2f}</li>
</ul>
</li>
""")
        
        return f"""
<div class='banking-response'>
<h2>Card Information</h2>
<p>Here are your card details, {user['name']}:</p>
<ul>
{"".join(cards_info)}
</ul>
<p><em>For security, never share your full card details with anyone.</em></p>

<h3>Card Services</h3>
<ul>
<li>Block/lost card</li>
<li>Increase credit limit</li>
<li>Transaction disputes</li>
<li>PIN regeneration</li>
</ul>
</div>
"""
    
    # Loan inquiries
    if any(token.text in ["loan", "emi", "borrow"] for token in doc):
        return f"""
<div class='banking-response'>
<h2>Loan Information</h2>
<p>Here are your loan details, {user['name']}:</p>

<h3>🏠 Home Loan</h3>
<ul>
<li>Principal Amount: ₹{user['loans']['home']['amount']:,.2f}</li>
<li>Outstanding: ₹{user['loans']['home']['remaining']:,.2f}</li>
<li>Monthly EMI: ₹{user['loans']['home']['emi']:,.2f}</li>
<li>Interest Rate: {user['loans']['home']['interest_rate']}</li>
</ul>

<h3>💼 Personal Loan</h3>
<ul>
<li>Principal Amount: ₹{user['loans']['personal']['amount']:,.2f}</li>
<li>Outstanding: ₹{user['loans']['personal']['remaining']:,.2f}</li>
<li>Monthly EMI: ₹{user['loans']['personal']['emi']:,.2f}</li>
<li>Interest Rate: {user['loans']['personal']['interest_rate']}</li>
</ul>

<p>📊 <strong>Credit Score</strong>: {user['credit_score']} (Excellent)</p>

<h3>Loan Services</h3>
<ul>
<li>EMI holiday</li>
<li>Foreclosure options</li>
<li>Loan restructuring</li>
<li>Top-up loan</li>
</ul>
</div>
"""
    
    # Fund transfer
    if any(token.text in ["transfer", "send money", "pay"] for token in doc):
        return """
<div class='banking-response'>
<h2>Fund Transfer</h2>
<p>You can transfer funds using these options:</p>

<h3>Quick Transfer</h3>
<ul>
<li>Between your own accounts</li>
<li>To saved beneficiaries</li>
<li>UPI payments</li>
</ul>

<h3>New Transfer</h3>
<ul>
<li>IMPS (Instant)</li>
<li>NEFT (Next working day)</li>
<li>RTGS (Large amounts)</li>
</ul>

<p><em>Daily transfer limit: ₹200,000</em></p>
</div>
"""
    
    return None

def get_general_response(query: str, history: List[Dict]) -> str:
    """Robust general conversation handler"""
    try:
        # Try banking model first if loaded
        if 'banking_model' in st.session_state and query.strip():
            try:
                inputs = st.session_state.banking_tokenizer.encode(
                    query + st.session_state.banking_tokenizer.eos_token,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True
                )
                
                if inputs.shape[1] > 0:
                    outputs = st.session_state.banking_model.generate(
                        inputs,
                        max_length=200,
                        pad_token_id=st.session_state.banking_tokenizer.eos_token_id,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.7
                    )
                    response = st.session_state.banking_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if response.strip():
                        return response
            except Exception as e:
                print(f"Banking model error: {e}")
        
        # Fallback to conversation model
        conv_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" 
             for msg in history[-3:] 
             if isinstance(msg, dict) and 'role' in msg and 'content' in msg]
        ) if history else ""
        
        prompt_text = f"Conversation history:\n{conv_history}\nUser: {query}\nAssistant:" if conv_history else f"User: {query}\nAssistant:"
        
        if len(prompt_text) > 10:
            try:
                result = conversation_model(
                    prompt_text,
                    max_length=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                if result and isinstance(result, list) and len(result) > 0:
                    return result[0]['generated_text']
            except Exception as e:
                print(f"Conversation model error: {e}")
        
        return "I couldn't process that request. Could you please rephrase or ask something else?"
    
    except Exception as e:
        print(f"General error in get_general_response: {e}")
        return "I'm having trouble understanding. Could you try asking differently?"

def get_time_of_day() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    return "night"

# Initialize app
if 'models' not in st.session_state:
    st.session_state.models = load_models()
    if st.session_state.models is None:
        st.stop()

nlp = st.session_state.models['nlp']
conversation_model = st.session_state.models['conversation_model']

# App Header with better UI
st.markdown("""
<div style="background: linear-gradient(135deg, #0078D4 0%, #004E8C 100%);
            color: white;
            padding: 1.8rem;
            border-radius: 12px;
            margin-bottom: 1.8rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <div style="display: flex; align-items: center; gap: 15px;">
        <div style="background: white; padding: 8px; border-radius: 50%;">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" fill="#0078D4"/>
                <path d="M12 6c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 10c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4z" fill="#0078D4"/>
                <circle cx="12" cy="12" r="2" fill="#0078D4"/>
            </svg>
        </div>
        <div>
            <h1 style="margin:0; font-size: 28px;">NeoBank AI Assistant</h1>
            <p style="margin:0; opacity:0.8; font-size: 14px;">Your intelligent banking companion</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": """
👋 Hello there! I'm Neo, your AI banking assistant. I can help you with:

• Checking account balances 💰  
• Reviewing transactions 📊  
• Loan applications 🏦  
• Credit card services 💳  
• Fund transfers 📤  
• Investment advice 📈  

What would you like to do today?
"""
    }]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "<div class='banking-response'>" in message["content"]:
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input handler
if prompt := st.chat_input("Type your message here..."):
    if not prompt.strip():
        st.warning("Please enter a valid message")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.spinner("Thinking..."):
        try:
            # First try banking-specific response
            banking_response = get_banking_response(prompt)
            if banking_response:
                response = banking_response
            else:
                # Check if we need banking model
                if any(word in prompt.lower() for word in ["transfer", "loan", "card"]):
                    if not load_banking_model():
                        response = "Banking features are currently unavailable"
                    else:
                        # Use banking model
                        inputs = st.session_state.banking_tokenizer.encode(
                            prompt + st.session_state.banking_tokenizer.eos_token,
                            return_tensors='pt',
                            max_length=512,
                            truncation=True
                        )
                        outputs = st.session_state.banking_model.generate(
                            inputs,
                            max_length=200,
                            pad_token_id=st.session_state.banking_tokenizer.eos_token_id,
                            do_sample=True,
                            top_p=0.95,
                            temperature=0.7
                        )
                        response = st.session_state.banking_tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # Fall back to general conversation
                    response = get_general_response(prompt, st.session_state.messages)
            
            # Personalize responses
            user = user_accounts["user123"]
            if any(word in prompt.lower() for word in ["thank", "thanks", "appreciate"]):
                response = random.choice([
                    f"You're welcome, {user['name']}! 😊",
                    f"Happy to help, {user['name']}!",
                    f"My pleasure, {user['name']}! Is there anything else I can assist you with?"
                ])
            elif any(word in prompt.lower() for word in ["hi", "hello", "hey"]):
                response = random.choice([
                    f"Hello {user['name']}! How can I assist you with your banking today?",
                    f"Hi there {user['name']}! What banking service can I help you with?",
                    f"Good {get_time_of_day()}, {user['name']}! How may I assist you?"
                ])
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                if "<div class='banking-response'>" in response:
                    st.markdown(response, unsafe_allow_html=True)
                else:
                    stream_response(response)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sorry, I'm having some technical difficulties. Please try again later."
            })

# Sidebar with quick actions
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=NeoBank+Logo", width=200)
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("💰 Check Balance"):
        st.session_state.messages.append({
            "role": "user",
            "content": "What's my account balance?"
        })
        st.rerun()
    
    if st.button("📊 View Transactions"):
        st.session_state.messages.append({
            "role": "user",
            "content": "Show me my recent transactions"
        })
        st.rerun()
    
    if st.button("💳 Card Services"):
        st.session_state.messages.append({
            "role": "user",
            "content": "Tell me about my credit card"
        })
        st.rerun()
    
    if st.button("📤 Transfer Funds"):
        st.session_state.messages.append({
            "role": "user",
            "content": "I want to transfer money"
        })
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📞 Customer Support")
    st.markdown("**24/7 Banking Support**  \n📞 1800-123-NEOB")
    st.markdown("**Email Assistance**  \n✉️ support@neobank.com")
    st.markdown("**Branch Locator**  \n📍 [Find nearest branch](#)")