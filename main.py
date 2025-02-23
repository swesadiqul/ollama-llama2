from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Ollama client
client = ollama.Client()

# Define request schema
class QueryRequest(BaseModel):
    prompt: str

# Define response schema
class QueryResponse(BaseModel):
    response: str

@app.post("/generate", response_model=QueryResponse)
async def generate_text(request: QueryRequest):
    """
    Endpoint to generate text using the Ollama client.
    """
    try:
        # Send the query to the model
        response = client.generate(model="mistral", prompt=request.prompt)
        return QueryResponse(response=response.response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")