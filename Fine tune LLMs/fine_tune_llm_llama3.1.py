import subprocess
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import json

os.environ['NCCL_P2P_DISABLE']="1"
os.environ['NCCL_IB_DISABLE']="1"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_acc(test_output_json):
    with open(test_output_json, mode='r', encoding='utf-8') as f:
        test_output = f.readlines()
    test_output = [json.loads(i) for i in test_output]
    label_list = [i['label'] for i in test_output]
    predict_list = [i['predict'] for i in test_output]
    recall=recall_score(label_list, predict_list, average='macro')
    precision=precision_score(label_list, predict_list, average='macro')
    f1=f1_score(label_list, predict_list, average='macro')
    accuracy=accuracy_score(label_list, predict_list)
    return accuracy

seed_list=[1,3,9,16,22,25,31,36,40,44,42,48,50,55,60,66,70,77,80,88]

asr_model_name_list=['wav2vec2-base-100h',
            'wav2vec2-base-960h',
            'wav2vec2-large-960h',
            'wav2vec2-large-960h-lv60',
            'wav2vec2-large-960h-lv60-self',
            'wav2vec2-large-xlsr-53-english',
            'wav2vec2-xls-r-1b-english',
            'hubert-large-ls960-ft',
            'hubert-xlarge-ls960-ft',
            'wavlm-libri-clean-100h-base-plus',
            'wavlm-libri-clean-100h-large',
            'whisper-tiny','whisper-base','whisper-small','whisper-medium','whisper-large','whisper-large-v2','whisper-large-v3']
asr_model_version_list=['fine_tune','original']

manual_acc_result=[]
asr_acc_result={}
for asr_model_name in asr_model_name_list:
    for version in asr_model_version_list:
        asr_acc_result[f'{asr_model_name}_{version}']=[]
        
manual_train_dataset='manual_train_data'
manual_test_dataset='manual_test_data'

for seed in seed_list:
    seed_everything(seed)
    #manual
    manual_train_output_dir=f'saves/manual/Meta-Llama-3.1-8B-Instruct/{seed}/'
    Path(manual_train_output_dir).mkdir(parents=True, exist_ok=True)
    manual_test_output_dir=f'predict/manual/Meta-Llama-3.1-8B-Instruct/{seed}/'
    Path(manual_test_output_dir).mkdir(parents=True, exist_ok=True)
    manual_test_output_json=f'{manual_test_output_dir}generated_predictions.jsonl'
    
    manual_train_command='''FORCE_TORCHRUN=1 llamafactory-cli train \
    --deepspeed examples/deepspeed/ds_z0_config.json \
    --stage sft \
    --do_train True \
    --model_name_or_path projects/huggingface_model/Meta-Llama-3.1-8B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --dataset {} \
    --overwrite_cache True \
    --cutoff_len 2048 \
    --learning_rate 0.0001 \
    --num_train_epochs 10.0 \
    --max_samples 1000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 5 \
    --save_steps 500 \
    --output_dir {} \
    --bf16 True \
    --plot_loss True \
    --overwrite_output_dir True \
    --ddp_timeout 180000000 \
    --lora_target all \
    --val_size 0.005 \
    --eval_strategy steps \
    --eval_steps 500 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 1'''.format(manual_train_dataset, manual_train_output_dir)
    
    os.system(manual_train_command)
    
    manual_test_command='''llamafactory-cli train \
    --stage sft \
    --model_name_or_path projects/huggingface_model/Meta-Llama-3.1-8B-Instruct \
    --adapter_name_or_path {} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template llama3 \
    --eval_dataset {} \
    --cutoff_len 2048 \
    --overwrite_cache True \
    --max_samples 100000 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate True \
    --max_new_tokens 2048 \
    --output_dir {} \
    --overwrite_output_dir True \
    --ddp_timeout 180000000 \
    --do_predict True'''.format(manual_train_output_dir, manual_test_dataset, manual_test_output_dir)
    
    os.system(manual_test_command)
    
    manual_acc_seed=cal_acc(manual_test_output_json)
    manual_acc_result.append(manual_acc_seed)
    mean_manual_acc=np.mean(manual_acc_result)
    std_manual_acc=np.std(manual_acc_result)
    print(f"manual:{seed}:{manual_acc_seed},all:{manual_acc_result},mean(std):{round(mean_manual_acc*100,2)}({round(std_manual_acc*100,2)})")
    with open('predict/manual/Meta-Llama-3.1-8B-Instruct/acc_result.txt', 'a+') as f_manual_acc:
        f_manual_acc.write(f"{seed}:{manual_acc_seed},all:{manual_acc_result},mean(std):{round(mean_manual_acc*100,2)}({round(std_manual_acc*100,2)})\n")
    
    for asr_model_name in asr_model_name_list:
        for version in asr_model_version_list:
            asr_train_dataset=f'{asr_model_name}_{version}_train_data'
            asr_test_dataset=f'{asr_model_name}_{version}_test_data'
            asr_train_output_dir=f'saves/asr/Meta-Llama-3.1-8B-Instruct/{asr_model_name}/{version}/{seed}/'
            Path(asr_train_output_dir).mkdir(parents=True, exist_ok=True)
            asr_test_output_dir=f'predict/asr/Meta-Llama-3.1-8B-Instruct/{asr_model_name}/{version}/{seed}/'
            Path(asr_test_output_dir).mkdir(parents=True, exist_ok=True)
            asr_test_output_json=f'{asr_test_output_dir}generated_predictions.jsonl'
            
            asr_train_command='''FORCE_TORCHRUN=1 llamafactory-cli train \
            --deepspeed examples/deepspeed/ds_z0_config.json \
            --stage sft \
            --do_train True \
            --model_name_or_path projects/huggingface_model/Meta-Llama-3.1-8B-Instruct \
            --preprocessing_num_workers 16 \
            --finetuning_type lora \
            --template llama3 \
            --dataset {} \
            --overwrite_cache True \
            --cutoff_len 2048 \
            --learning_rate 0.0001 \
            --num_train_epochs 10.0 \
            --max_samples 1000 \
            --per_device_train_batch_size 2 \
            --gradient_accumulation_steps 2 \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --logging_steps 5 \
            --save_steps 500 \
            --output_dir {} \
            --bf16 True \
            --plot_loss True \
            --overwrite_output_dir True \
            --ddp_timeout 180000000 \
            --lora_target all \
            --val_size 0.005 \
            --eval_strategy steps \
            --eval_steps 500 \
            --preprocessing_num_workers 16 \
            --per_device_eval_batch_size 1'''.format(asr_train_dataset, asr_train_output_dir)
            
            os.system(asr_train_command)

            asr_test_command='''llamafactory-cli train \
            --stage sft \
            --model_name_or_path projects/huggingface_model/Meta-Llama-3.1-8B-Instruct \
            --adapter_name_or_path {} \
            --preprocessing_num_workers 16 \
            --finetuning_type lora \
            --template llama3 \
            --eval_dataset {} \
            --cutoff_len 2048 \
            --overwrite_cache True \
            --max_samples 100000 \
            --per_device_eval_batch_size 4 \
            --predict_with_generate True \
            --max_new_tokens 2048 \
            --output_dir {} \
            --overwrite_output_dir True \
            --ddp_timeout 180000000 \
            --do_predict True'''.format(asr_train_output_dir, asr_test_dataset, asr_test_output_dir)
            os.system(asr_test_command)
            
            asr_acc_seed=cal_acc(asr_test_output_json)
            asr_acc_result[f'{asr_model_name}_{version}'].append(asr_acc_seed)
            asr_acc_result_all=asr_acc_result[f'{asr_model_name}_{version}']
            mean_asr_acc=np.mean(asr_acc_result_all)
            std_asr_acc=np.std(asr_acc_result_all)
            print(f"asr:{seed}:{asr_acc_seed},all:{asr_acc_result_all},mean(std):{round(mean_asr_acc*100,2)}({round(std_asr_acc*100,2)})")
            with open(f'predict/asr/Meta-Llama-3.1-8B-Instruct/{asr_model_name}/{version}/acc_result.txt', 'a+') as f_asr_acc:
                f_asr_acc.write(f"{seed}:{asr_acc_seed},all:{asr_acc_result_all},mean(std):{round(mean_asr_acc*100,2)}({round(std_asr_acc*100,2)})\n")