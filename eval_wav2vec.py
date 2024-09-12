import os
import configparser
import logging
import shutil
import sys
import gc
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
import pandas as pd
import soundfile as sf
import numpy as np
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import Wav2Vec2Processor, AutoModelForCTC
from evaluate import load
from tqdm import tqdm
from pathlib import Path


def speech_file_to_array_fn(batch):
    """
    batchfy and map the wav file to arrays

    :param batch: _description_
    :type batch: _type_
    :return: _description_
    :rtype: _type_
    """
    speech_array, sampling_rate = sf.read(batch["locs"])
    logging.info(f"Processing {batch['locs']}")
    batch["speech"] = speech_array
    
    batch["sampling_rate"] = sampling_rate
    batch["locs"] = batch["locs"]
    return batch

def asr_process_dataset(dataset,processor,model,f):
    pred_str_list=[]
    for i in tqdm(range(len(dataset['dataset']))):
        input_values=processor(dataset['dataset'][i]["speech"], sampling_rate=16000,return_tensors="pt",padding="longest").input_values.to("cuda")
        with torch.no_grad():
             pred_logits = model(input_values).logits
        pred_ids = torch.argmax(pred_logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        pred_str_list.append(pred_str)
        line=f"{dataset['dataset'][i]['locs'].split('/')[-1]},pre_str:{pred_str},label_str:{dataset['dataset'][i]['trans']}"
        f.write(line+'\n')
    wer = wer_metric.compute(predictions=pred_str_list, references=dataset['dataset']['trans'])
    cer = cer_metric.compute(predictions=pred_str_list, references=dataset['dataset']['trans'])
    result=f"wer: {round(wer, 4)}, cer: {round(cer, 4)}"
    f.write(result+'\n')
    f.close()
    return wer,cer,pred_str_list

def save_asr_trans(dataset,pred_str_list,f):
    trans_dic={p:[] for p in set([q.split('/')[-1].split('_')[0] for q in dataset['dataset']['locs']])}
    for i in range(len(dataset['dataset'])):
        wav_name=dataset['dataset'][i]['locs'].split('/')[-1].split('_')[0]
        trans_dic[wav_name].append(pred_str_list[i])
    for key in trans_dic.keys():
        trans_dic[key]='. '.join(trans_dic[key])
        f.write(f'{key}:{trans_dic[key]}\n')
    f.close()
        
wer_metric = load("projects/utils/evaluate-0.4.2/metrics/wer", experiment_id="3")
cer_metric = load("projects/utils/evaluate-0.4.2/metrics/cer", experiment_id="3")
result_dir='projects/ASR_error/paradox-asr-main/asr_wer_result_english/'
dataset_dict={"asr_test_dataset":"projects/ASR_error/data/test/audio_chunk/test.csv",
              'ADReSS_test_dataset':"projects/ASR_error/data/ADReSS/test/audio_chunk/ADReSS_test.csv",
              'ADReSS_train_dataset':"projects/ASR_error/data/ADReSS/train/audio_chunk/ADReSS_train.csv",
              'ADReSS_cc_dataset':"projects/ASR_error/data/ADReSS/cc/audio_chunk/ADReSS_cc.csv",
              'ADReSS_cd_dataset':"projects/ASR_error/data/ADReSS/cd/audio_chunk/ADReSS_cd.csv",
              'ADReSS_all_dataset':"projects/ASR_error/data/ADReSS/audio_chunk/ADReSS_all.csv"}

for key in dataset_dict.keys():
    dataset_dict[key]= load_dataset("csv",data_files={"dataset": dataset_dict[key]})
    dataset_dict[key] = dataset_dict[key].map(speech_file_to_array_fn, num_proc=32)

fine_tune_model_dir='projects/ASR_error/paradox-asr-main/ft-models/asr/'
original_model_dir='projects/huggingface_model/'
model_name_list = [p.name for p in Path(fine_tune_model_dir).iterdir() if p.is_dir() if 'whisper' not in p.name]
f_wer_result=open(f'{result_dir}wav2vec_wer_result.txt','w')
for model_name in tqdm(model_name_list):
    fine_tune_model_path=os.path.join(fine_tune_model_dir,model_name)
    min_checkpoints = str(min([int(p.name.split('-')[-1]) for p in Path(fine_tune_model_path).iterdir() if p.is_dir()]))
    fine_tune_model_path=os.path.join(fine_tune_model_path,f'checkpoint-{min_checkpoints}')
    print(fine_tune_model_path)
    original_model_path=os.path.join(original_model_dir,model_name)
    print(original_model_path)
    
    fine_tune_model=AutoModelForCTC.from_pretrained(fine_tune_model_path).to("cuda")
    original_model=AutoModelForCTC.from_pretrained(original_model_path).to("cuda")
    model_dict={'fine_tune_model':fine_tune_model,'original_model':original_model}
    processor=Wav2Vec2Processor.from_pretrained(original_model_path)
    
    original_model.eval()
    fine_tune_model.eval()
    
    for dataset_name in dataset_dict.keys():
        dataset=dataset_dict[dataset_name]
        for model_key in model_dict.keys():
            model=model_dict[model_key]             
            print(f"Processing {dataset_name} with {model_key} {model_name}")
            f_utt=open(f'{result_dir}{model_key}_{model_name}_{dataset_name}.txt','w')
            f_par=open(f'{result_dir}{model_key}_{model_name}_{dataset_name}_trans.txt','w')
            wer,cer,pred_str_list=asr_process_dataset(dataset,processor,model,f_utt)
            f_wer_result.write(f"{model_key}_{model_name}_{dataset_name},wer:{wer},cer:{cer}\n")
            save_asr_trans(dataset,pred_str_list,f_par)
f_wer_result.close()          
            
