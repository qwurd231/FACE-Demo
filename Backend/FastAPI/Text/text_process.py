from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException
import random

router = APIRouter()

class Text(BaseModel):
    text: Optional[str] = None
    frequency: Optional[list] = None
    spectra: Optional[list] = None

@router.post("/text" , response_model=Text)
def text(data: Text):
    if data.text == '' or data.text is None:
        raise HTTPException(status_code=400)
    else:
        print(data.text)
        frequency = [random.random() for _ in range(15)].sort()
        spectra = sorted([random.randint(2, 10) for _ in range(15)], reverse=True)
        print(f'frequency: {frequency}')
        print('spectra:', spectra)

        return {"frequency": frequency, "text": data.text + "  Well, hello to you too!",  "spectra": spectra}