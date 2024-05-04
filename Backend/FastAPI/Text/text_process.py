"""
    Module for text processing.
    This module is used to process the text data from the frontend.
    The text data is split into paragraphs and each paragraph is assigned a color.
    The color is used to represent the sentiment of the paragraph.
    The text data is also processed to generate the frequency and spectra data.
    The frequency and spectra data is used to generate the graph.
"""

import gzip
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.requests import Request

router = APIRouter()

class Text(BaseModel):
    """Model for text data."""	

    text: Optional[dict | str] = None

class Result(BaseModel):
    """
        Model for the response data.
        The response data contains the text data, color data, frequency data, and spectra data.
    """

    text: Optional[dict] = None

def slide_window(fre, spe, window_size, step_size):
    """Slide window to get the frequency and spectra data."""

    result_fre = []
    result_spe = []
    for i in range(0, len(spe) - window_size + 1, step_size):
        result_fre.append(fre[i + round(window_size / 2)])
        result_spe.append(sum(spe[i : i + window_size]) / window_size)
        if i <= 15:
            print("spe[i : i + window_size]: ", spe[i : i + window_size])
            print("sum(spe[i : i + window_size]): ", sum(spe[i : i + window_size]))
            print("spe[i: i + window_size]: ", spe[i: i + window_size])
            print("result_spe: ", result_spe)
    return result_fre, result_spe


def get_valid_text(data, request: Request):
    """Get the valid text data."""

    original_text = ''
    print(request.headers)
    if request.headers['content-encoding'] != 'gzip':
        original_text = data.text
    else:
        print(data.text)
        bytes_text = bytes([data.text[str(i)] for i in range(len(data.text))])
        print('bytes_text: ', bytes_text)
        original_text = gzip.decompress(bytes_text).decode('utf-8')
    return original_text



@router.post("/text")
def text(data: Text, request: Request):
    """Process the text data."""

    original_text = ''
    print(request.headers)
    if 'content-encoding' not in request.headers \
    or 'gzip' not in request.headers['content-encoding']:
        original_text = data.text
    else:
        print(data.text)
        bytes_text = bytes([data.text[str(i)] for i in range(len(data.text))])
        print('bytes_text: ', bytes_text)
        original_text = gzip.decompress(bytes_text).decode('utf-8')

    if original_text != '' and original_text is not None:
        print(original_text)
        if original_text.find('\n\n') == -1:
            valid_text = [original_text]
        else:
            valid_text = original_text.split('\n\n')
        print("valid_text: ", valid_text)
        color = ['#B7B94B' for _ in range(len(valid_text))]

        frequency = []
        spectra = []
        with open('..\\FACE\\FACE-main\\data\\demo_human.fft.txt', 'r', encoding='UTF-8') as fr:
            for i, line in enumerate(fr):
                if i == 0:
                    continue
                if i >= 2 and float(line.split(',')[0]) < frequency[-1]:
                    break
                if i >= 2:
                    frequency.append(float(line.split(',')[0]))
                    spectra.append(float(line.split(',')[1]))
                else:
                    frequency.append(float(line.split(',')[0]))
                    spectra.append(float(line.split(',')[1]))

        print("frequency", frequency)
        print("spectra", spectra)
        slide_frequency, slide_spectra = slide_window(frequency, spectra, 20, 5)
        # from list to string
        print(valid_text)
        valid_text = '<comma/>'.join(valid_text)
        print(valid_text)
        print(color)
        color = ','.join(color)
        print(color)
        slide_frequency = ','.join([str(i) for i in slide_frequency])
        print(slide_frequency)
        slide_spectra = ','.join([str(i) for i in slide_spectra])
        print(slide_spectra)

        s = "text:" + valid_text + "</>color:" + color + "</>frequency:" \
            + slide_frequency + "</>spectra:" + slide_spectra
        print(s)
        print(len(s))
        result = gzip.compress(s.encode('utf-8'))
        result_dict = {str(i): byte for i, byte in enumerate(result)}
        print(result_dict)
        print(len(result_dict))

        # result_text = gzip.compress(json.dumps(valid_text).encode('utf-8'))
        # result_text_dict = {str(i): byte for i, byte in enumerate(result_text)}
        # print(result_text_dict)

        # result_color = gzip.compress(json.dumps(color).encode('utf-8'))
        # result_color_dict = {str(i): byte for i, byte in enumerate(result_color)}

        # result_frequency = gzip.compress(json.dumps(slide_frequency).encode('utf-8'))
        # result_frequency_dict = {str(i): byte for i, byte in enumerate(result_frequency)}

        # result_spectra = gzip.compress(json.dumps(slide_spectra).encode('utf-8'))
        # result_spectra_dict = {str(i): byte for i, byte in enumerate(result_spectra)}

        # response = Result(text=result_text_dict, color=result_color_dict,
        #                 frequency=result_frequency_dict, spectra=result_spectra_dict)

        return result_dict
    else:
        raise HTTPException(status_code=400, detail="Invalid text data.")
