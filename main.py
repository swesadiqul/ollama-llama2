from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

# Initialize the FastAPI app
app = FastAPI()

# Initialize the Ollama client
client = ollama.Client()

# Define request schema
class QueryRequest(BaseModel):
    model: str
    prompt: str

# Define response schema
class QueryResponse(BaseModel):
    response: str

@app.post("/generate", response_model=QueryResponse)
async def generate_text(request: QueryRequest):
    """
    Endpoint to generate text using the Ollama client.

    Parameters:
    - model: The name of the model to use.
    - prompt: The input prompt for the model.

    Returns:
    - The model's response as a JSON object.
    """
    try:
        # Send the query to the model
        response = client.generate(model=request.model, prompt=request.prompt)
        return QueryResponse(response=response.response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
