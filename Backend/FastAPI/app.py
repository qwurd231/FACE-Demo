from Text.text import router as text_router
from fastapi import FastAPI
import uvicorn
from Text.text import router as text_router

app = FastAPI()

# Include the router from Text/text.py
app.include_router(text_router, tags=["Text"])

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)