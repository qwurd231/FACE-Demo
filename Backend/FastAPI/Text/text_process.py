from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException
from starlette.requests import Request
import json
import gzip

router = APIRouter()

class Text(BaseModel):
    text: Optional[object] = None

class Result(BaseModel):
    text: Optional[str] = None
    frequency: Optional[list] = None
    spectra: Optional[list] = None


@router.post("/text" 
             #, response_model=Result)
                )
def text(data: Text, request: Request):
    original_text = ''
    print(request.headers)
    if request.headers['content-encoding'] != 'gzip':
        original_text = data.text
    else:
        print(data.text)
        bytes_text = bytes([data.text[str(i)] for i in range(len(data.text))])
        print('bytes_text: ', bytes_text)
        original_text = gzip.decompress(bytes_text).decode('utf-8')
        
    if original_text == '' or original_text is None:
        raise HTTPException(status_code=400)
    else:
        print(original_text)
        frequency = []
        spectra = []
        with open('..\\FACE\\FACE-main\\data\\demo_human.fft.txt', 'r', encoding='UTF-8') as fr:
            for i, line in enumerate(fr):
                if i == 0: continue
                if i >= 2 and float(line.split(',')[0]) < frequency[-1]: break
                if i >= 2:
                    frequency.append(float(line.split(',')[0]))
                    spectra.append(float(line.split(',')[1]))
                else:
                    frequency.append(float(line.split(',')[0]))
                    spectra.append(float(line.split(',')[1]))

        
        result_text = gzip.compress((original_text + 'hh').encode('utf-8'))
        result_text_dict = {str(i): byte for i, byte in enumerate(result_text)}

        result_frequency = gzip.compress(json.dumps(frequency).encode('utf-8'))
        result_frequency_dict = {str(i): byte for i, byte in enumerate(result_frequency)}

        result_spectra = gzip.compress(json.dumps(spectra).encode('utf-8'))
        result_spectra_dict = {str(i): byte for i, byte in enumerate(result_spectra)}

        response = {"text": result_text_dict, "frequency": result_frequency_dict, "spectra": result_spectra_dict}

        # response = {"text": result_text_dict, "frequency": frequency, "spectra": spectra}


        return response