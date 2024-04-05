from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException

router = APIRouter()

class Text(BaseModel):
    text: Optional[str] = None

@router.post("/text" , response_model=Text)
def text(data: Text):
    if data.text is None:
        raise HTTPException(status_code=400, detail="Text is required")
    else:
        print(data.text)
        return {"text": data.text + "000000avbearrsvwrsv"}
    
@router.get("/select", response_model=Text)
def select():
    return {"text": "000000"}