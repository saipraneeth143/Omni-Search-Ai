from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="OmniSearch AI Backend", version="1.0")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the OmniSearch AI API. System is running."}

@app.post("/api/chat")
def chat_with_data(request: QueryRequest):
    # This is a placeholder for the RAG/LLM logic
    # In the future, this will connect to Pinecone/Vector DB and OpenAI
    
    mock_response = {
        "question": request.question,
        "answer": "This is an AI-generated answer retrieved from your fragmented documents.",
        "citations": ["HR_Handbook.pdf - Page 4", "Company_Policy_Notion_Page"]
    }
    return mock_response
