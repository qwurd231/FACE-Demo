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

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch import nn
from tqdm import tqdm
from einops import rearrange
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift

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


def load_model():
    """Load the model for text processing."""

    model_path = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def process(model, tokenizer, text):
    """Process the text data."""

    nll_loss_list = []
    device = model.device
    criterian = nn.NLLLoss(reduction='none')
    log_softmax = nn.LogSoftmax(dim=1)
    
    for line in tqdm(text):
        print("line: ", line)
        encoded_input = tokenizer(line,
                                  return_tensors='pt',
                                  padding=True,
                                  truncation=True,
                                  max_length=1024)
        input_ids = encoded_input['input_ids']

        try:
            output = model(**encoded_input, labels=input_ids)
        except Exception:
            print('line:', line)
            print('input_ids:', input_ids)
            raise
        logits = output.logits.to(device)
        target = encoded_input['input_ids'].to(device)

        logits = rearrange(logits, 'B L V -> B V L')
        shift_logits = logits[
            ..., :, :-1]  # Use the first L-1 tokens to predict the next
        shift_target = target[..., 1:]

        nll_loss = criterian(log_softmax(shift_logits),
                                shift_target).squeeze()
        res = nll_loss.tolist()
        if not isinstance(res, list):
            res = [res]

        try:
            res_str = ' '.join(f'{num:.4f}' for num in res)
        except Exception:
            print('line:', line)
            print('input_ids:', input_ids)
            print('logits.shape:', logits.shape)
            print('res:', res)
            raise
        else:
            nll_loss_list.append(res_str)

    return nll_loss_list


def fft_pipeline(text, n_samples=np.inf, normalize=False):
    """Compute the frequency and spectra data."""

    data_list = _read_data(text)

    if n_samples < np.inf:
        data = [np.asarray(d) for d in data_list[:n_samples]]
    else:
        data = [np.asarray(d) for d in data_list]
    if normalize:
        data_norm = []
        epsion = 1e-6
        for d in data:
            d_mean = np.mean(d)
            sd_mean = np.std(d)
            d_norm = (d - d_mean) / (sd_mean + epsion)
            data_norm.append(d_norm)
        data = data_norm

    freqs, powers = compute_fft(data)

    df = pd.DataFrame.from_dict({
        'freq': np.concatenate(freqs),
        'power': np.concatenate(powers)
    })
    return df


def compute_fft(data):
    """Compute the frequency and spectra data."""

    freqs, powers = [], []
    print(data)
    print(len(data))
    for i in tqdm(range(len(data))):
        x = data[i]
        try:
            N = x.shape[-1]
            freq_x = fftshift(fftfreq(N))
            sp_x = fftshift(fft(x)).real  # take the real part
        except Exception:
            print(f'Error in sample {i}: {x}')
            raise
        freqs.append(freq_x[len(freq_x) // 2:])
        powers.append(sp_x[len(sp_x) // 2:])
    return freqs, powers


def _read_data(text, N=np.inf):
    data = []
    count = 0
    for line in text:
        line = line.strip()
        if line == '':
            continue
        num = list(map(float, line.split()))
        data.append(num)
        count += 1
        if count >= N:
            break
    return data


@router.post("/text")
def text(data: Text, request: Request):
    """Process the text data."""

    original_text = ''
    print(request.headers)
    if 'content-encoding' not in request.headers \
    or 'gzip' not in request.headers['content-encoding']:
        original_text = data.text
    else:
        # print(data.text)
        bytes_text = bytes([data.text[str(i)] for i in range(len(data.text))])
        # print('bytes_text: ', bytes_text)
        original_text = gzip.decompress(bytes_text).decode('utf-8')

    if original_text != '' and original_text is not None:
        # print(original_text)
        if original_text.find('\n\n') == -1:
            valid_text = [original_text.strip()]
        else:
            valid_text = [i.strip() for i in original_text.split('\n\n') if i.strip() != '']
        print("valid_text: ", valid_text)
        color = ['#B7B94B' for _ in range(len(valid_text))]

        model, tokenizer = load_model()
        nll_loss = process(model, tokenizer, valid_text)
        print("nll_loss: ", nll_loss)

        df = fft_pipeline(nll_loss, n_samples=np.inf, normalize=False)
        print("df: ", df)
        print("df['freq']: ", df['freq'])
        print("df['power']: ", df['power'])

        frequency = df['freq'].tolist()
        spectra = df['power'].tolist()

        slide_frequency, slide_spectra = slide_window(frequency, spectra, 2, 1)
        # from list to string
        # print(valid_text)
        valid_text = 'の'.join(valid_text)
        # print(valid_text)
        # print(color)
        color = ','.join(color)
        # print(color)
        slide_frequency = ','.join([str(i) for i in slide_frequency])
        # print(slide_frequency)
        slide_spectra = ','.join([str(i) for i in slide_spectra])
        # print(slide_spectra)

        s = "text:" + valid_text + "</>color:" + color + "</>frequency:" \
            + slide_frequency + "</>spectra:" + slide_spectra
        # print(s)
        # print(len(s))
        result = gzip.compress(s.encode('utf-8'))
        result_dict = {str(i): byte for i, byte in enumerate(result)}
        # print(result_dict)
        # print(len(result_dict))

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
