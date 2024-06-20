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
from fastapi import APIRouter, HTTPException, Depends
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

from sqlite_db_setting import insert_data, select_data, create_connection, create_table, delete_old_data, delete_all_data
import time

from statsmodels.gam.api import GLMGam
from statsmodels.gam.smooth_basis import CubicSplines

router = APIRouter()

counter = True

def allowed_ip(request: Request):
    """Check if the request is allowed."""

    # if request.client.host in ip_list:
    #     raise HTTPException(status_code=429, detail="Too Many Requests, please wait for 60 seconds.")
    # return True
    conn = create_connection("sqlite.db")
    print("this is the connection", conn)
    create_table_sql = """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ip_address TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
    create_table(conn, create_table_sql)

    select_data_sql = f"SELECT * FROM users WHERE ip_address = '{request.client.host}'"
    rows = select_data(conn, select_data_sql)
    print(rows)
    global counter
    if not counter:
        counter = True
        raise HTTPException(status_code=429, detail="Too Many Requests, please wait for 60 seconds.")
    
    counter = False

    insert_data_sql = f"INSERT INTO users (ip_address) VALUES ('{request.client.host}')"
    insert_data(conn, insert_data_sql)

    return True 

def check_every_minute():
    conn = create_connection("sqlite.db")

    while True:
        delete_old_data(conn, "DELETE FROM users WHERE created_at < ?")
        print("2 seconds sleep")
        time.sleep(2)


class Text(BaseModel):
    """Model for text data."""	

    text: Optional[dict | str] = None

class Result(BaseModel):
    """
        Model for the response data.
        The response data contains the text data, color data, frequency data, and spectra data.
    """

    text: Optional[dict] = None

def generate_model_text(input_prompt: str):
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Encode text input
    print("Input text length:", len(input_prompt))
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
    print(input_ids)

    print('the length of input_ids:', len(input_ids[0]))
    index = 0
    final_text = ''

    while index < len(input_ids[0]):
        fixed_length_input_ids = input_ids[:, index:index+1000]
        print('the length of fixed_length_input_ids:', len(fixed_length_input_ids[0]))
        print('the shape of fixed_length_input_ids:', fixed_length_input_ids.shape)
        index += 1000

        # Generate text
        output = model.generate(fixed_length_input_ids, num_return_sequences=1, 
                                do_sample=True, attention_mask=None, no_repeat_ngram_size=2, 
                                top_k=50, top_p=0.95, temperature=0.7)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Append the generated text to the final text
        final_text += output_text

    print("Final text:", final_text)
    print("Final text length:", len(final_text))
    return final_text

def slide_window(fre, spe, window_size, step_size):
    """Slide window to get the frequency and spectra data."""

    result_fre = []
    result_spe = []
    for i in range(0, len(spe) - window_size + 1, step_size):
        result_fre.append(fre[i + round(window_size / 2)])
        result_spe.append(sum(spe[i : i + window_size]) / window_size)

    return result_fre, result_spe

def segment_fre_spe(fre, spe):
    """Segment the frequency and spectra data."""

    # split fre, spe whenever fre encounters 0.0
    fre_np = np.array(fre)
    spe_np = np.array(spe)

    split_idx = np.where(fre_np == 0.0)

    print("split_idx: ", split_idx)
    fre_list = np.split(fre_np, split_idx[0])[1:]
    spe_list = np.split(spe_np, split_idx[0])[1:]
    
    print(fre_list)
    print(spe_list)

    return fre_list, spe_list

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

def calculate_fre_spe(text: list):
    model, tokenizer = load_model()
    nll_loss = process(model, tokenizer, text)
    print("nll_loss: ", nll_loss)

    df = fft_pipeline(nll_loss, n_samples=np.inf, normalize=False)
    df.to_csv('df.csv', index=False)

    frequency = df['freq'].tolist()
    spectra = df['power'].tolist()
    frequency, spectra = segment_fre_spe(frequency, spectra)

    return frequency, spectra, df

def gam_graph(df: pd.DataFrame, index: int):
    if index == 1:
        x_spline = df['freq']
    else:
        x_spline = []
        for i in range(len(df['freq'])):
            x_spline.extend([df['freq'][i]])
        x_spline.sort()
    print('x_spline:', x_spline)
    cs = CubicSplines(x_spline, df=[20])
    if index == 1:
        gam_bs = GLMGam.from_formula('power ~ freq', data=df, smoother=cs)
    else:
        df['freq'] = x_spline
        df_power = []
        for i in range(len(df['power'])):
            df_power.extend([df['power'][i]])
        df['power'] = df_power.sort()
        gam_bs = GLMGam.from_formula('power ~ freq', data=df, smoother=cs)

    res_bs = gam_bs.fit()
    print('res_bs.fittedvalues:', res_bs.fittedvalues)
    print(res_bs.summary())

    return res_bs.fittedvalues

@router.post("/text", dependencies=[Depends(allowed_ip)])
def text(data: Text, request: Request):
    """Process the text data."""

    # conn = create_connection("sqlite.db")
    # insert_data_sql = f"INSERT INTO users (ip_address) VALUES ('{request.client.host}')"
    # insert_data(conn, insert_data_sql)
    original_text = ''
    print(request.headers)
    if 'content-encoding' not in request.headers \
    or 'gzip' not in request.headers['content-encoding']:
        original_text = data.text
    else:
        # print(data.text)
        bytes_text = bytes([data.text[str(i)] for i in range(len(data.text))])
        #bytes_text = np.array([data.text[str(i)] for i in range(len(data.text))], dtype=np.uint8).tobytes()
        # print('bytes_text: ', bytes_text)
        original_text = gzip.decompress(bytes_text).decode('utf-8')

    if original_text != '' and original_text is not None:
        if original_text.find('\n\n') == -1:
            valid_text = [original_text.strip()]
        else:
            valid_text = [i.strip() for i in original_text.split('\n\n') if i.strip() != '']
        print("valid_text: ", valid_text)

        input_prompt = "This is half of the text, please complete the other half: "
        input_text = input_prompt + ' '.join(valid_text[:len(valid_text) // 2])[0:-1]
        print("input_text: ", input_text)
        model_text = generate_model_text(input_text)
        
        color = ["{0}{1}{2}".format("#", hex(1096011 + (i + 1) * (12040523 - 1096011) // len(valid_text))[2:].upper(), "A0" )for i in range(len(valid_text))]

        frequency, spectra, df = calculate_fre_spe(valid_text)
        print("frequency: ", frequency)
        print("spectra: ", spectra)
        model_fre, model_spe, model_df = calculate_fre_spe([model_text])
        print("model_fre: ", model_fre)
        print("model_spe: ", model_spe)

        #text_fittedvalues = gam_graph(df, 0)
        #print("fittedvalues: ", text_fittedvalues)
        model_fittedvalues = gam_graph(model_df, 1)
        print("fittedvalues: ", model_fittedvalues)

        # slide_frequency, slide_spectra = slide_window(frequency, spectra, 2, 1)
        # from list to string
        valid_text = '分'.join(valid_text)
        color = ','.join(color)
        origin_join_frequency = 'る'.join([','.join([str(i) for i in f]) for f in frequency])
        origin_join_spectra = 'る'.join([','.join([str(i) for i in s]) for s in spectra])
        model_frequency = ','.join([str(i) for i in model_fre[0]])
        model_spectra = ','.join([str(i) for i in model_spe[0]])
        m_fittedvalues = ','.join([str(i) for i in model_fittedvalues])
        #t_fittedvalues = ','.join([str(i) for i in text_fittedvalues])
        
        print("origin_frequency:", origin_join_frequency)
        print("origin_spectra:", origin_join_spectra)
        print("model_frequency:", model_frequency)
        print("model_spectra:", model_spectra)
        print("model_fittedvalues:", model_fittedvalues)

        s = "text:" + valid_text + "けcolor:" + color + "けfrequency:" \
            + origin_join_frequency + "けspectra:" + origin_join_spectra + "けmodel text:" \
            + model_text + "けmodel frequency:" + model_frequency + "けmodel spectra:" + model_spectra \
            + "けmodel fittedvalues:" + m_fittedvalues + "けpropmt length:" + str(len(input_text) - len(input_prompt)) \
        #    + "けtexts fittedvalues:" + t_fittedvalues
        print(s)
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
    
if __name__ == "__main__":
    check_every_minute()