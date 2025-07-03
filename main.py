from fastapi import FastAPI
from pydantic import BaseModel
import spacy  # Using spaCy instead of Dialogflow

app = FastAPI()
nlp = spacy.load("en_core_web_md")  # Medium English model

class Query(BaseModel):
    query: str

def process_query(text: str) -> str:
    doc = nlp(text.lower())
    
    if any(token.text in ["balance", "money"] for token in doc):
        return "Your account balance is ₹25,000 as of today."
    
    elif any(token.text in ["loan", "borrow"] for token in doc):
        return "Loan applications take 2 business days to process."
    
    elif any(token.text in ["transaction", "history"] for token in doc):
        return "Your last 5 transactions: 1. Amazon ₹1,200 2. ATM ₹5,000"
    
    return "I can help with balances, loans, or transactions. Please clarify."

@app.post("/chat")
async def chat(query: Query):
    return {"response": process_query(query.query)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)