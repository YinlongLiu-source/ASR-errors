from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from datasets import load_dataset
import pandas as pd
bert_path='hf_model/bert-large-uncased'
roberta_path='hf_model/roberta-large'
llama2_path='hf_model/llama3.1'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModel.from_pretrained(bert_path).to(device)
bert_model.eval()

roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_path)
roberta_model = AutoModel.from_pretrained(roberta_path).to(device)
roberta_model.eval()

llama2_tokenizer = AutoTokenizer.from_pretrained(llama2_path)
llama2_tokenizer.pad_token = "[PAD]"
llama2_tokenizer.padding_side = "right"
config_kwargs = {
    "trust_remote_code": True,
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "output_hidden_states": True
}
llama2_model_config = AutoConfig.from_pretrained(llama2_path, **config_kwargs)
llama2_model = AutoModelForCausalLM.from_pretrained(
    llama2_path,
    trust_remote_code=True,
    config=llama2_model_config,
    device_map=device,
    torch_dtype=torch.float16)
llama2_model.eval()

dataset = load_dataset('csv',data_files={"train":'projects/LLMEmbed/dataset/train_manual.csv',"test":'projects/LLMEmbed/dataset/test_manual.csv'})

#print(dataset)
# for idx in range(len(labels)):
#     labels[idx] = torch.tensor(labels[idx])
# labels = torch.stack(labels)
# print(labels.shape)

data = pd.DataFrame({"text": dataset['test']["joined_all_par_trans"], "AD": dataset['test']["ad"]})
labels = dataset['test']['ad']
sents=data['text'][:1]
#print(sents)
# print(bert_model)
# print(roberta_model)
model_tokenizer_dict={"bert":(bert_model, bert_tokenizer), 
                      "roberta":(roberta_model, roberta_tokenizer), 
                      "llama3.1":(llama2_model, llama2_tokenizer)}

def extract_representations(model_tokenizer_dict, sents,max_length=512):

    #bert representation
    bert_sents_batch_encoding = model_tokenizer_dict['bert'][1]([s for s in sents], return_tensors='pt', max_length=max_length, padding="max_length", truncation=True).to(device)
    #print(bert_sents_batch_encoding)
    with torch.no_grad():
        #print(model_tokenizer_dict['bert'][0](**bert_sents_batch_encoding))
        bert_reps = model_tokenizer_dict['bert'][0](**bert_sents_batch_encoding).pooler_output
    #print(bert_reps)
    
    #roberta representation
    roberta_sents_batch_encoding = model_tokenizer_dict['roberta'][1]([s for s in sents], return_tensors='pt', max_length=max_length, padding="max_length", truncation=True).to(device)
    with torch.no_grad():
        roberta_reps = model_tokenizer_dict['roberta'][0](**roberta_sents_batch_encoding).last_hidden_state[:, 0, :]
    #print(roberta_reps)
        # batch_outputs = model_tokenizer_dict['roberta'][0](**roberta_sents_batch_encoding )
        # print(batch_outputs)
        # reps_batch = batch_outputs.last_hidden_state[:, 0, :]
        # print(batch_outputs[0][:,0,:])
        # print(batch_outputs[0][:,0])
        # print(reps_batch)
        # print(batch_outputs[0].shape)
        # print(model_tokenizer_dict['roberta'][0].pooler)
        # print(model_tokenizer_dict['roberta'][0].pooler(batch_outputs[0])[0,:5])
        # #print(model_tokenizer_dict['roberta'][0].pooler(batch_outputs[0][:,0]))
        # print(model_tokenizer_dict['roberta'][0].pooler(batch_outputs[0]).shape)
        # print(batch_outputs.pooler_output[0,:5])
        # print(batch_outputs[1])
        # print(reps_batch==batch_outputs.pooler_output)
        
        
    #llama2 representation
    llama2_sents_batch_encoding = model_tokenizer_dict['llama3'][1]([s for s in sents], return_tensors='pt', max_length=max_length, padding="max_length", truncation=True).to(device)
    #print(sents_batch_encoding)
    with torch.no_grad():
        llama2_batch_outputs =  model_tokenizer_dict['llama3'][0](**llama2_sents_batch_encoding)
        #print(batch_outputs)
        llama2_reps_batch_5L = []
        #print(batch_outputs.hidden_states[-1].shape)
        #print(torch.mean(batch_outputs.hidden_states[-1], axis=1).shape)
        #后5层的平均值
        for layer in range(-1, -6, -1):
            llama2_reps_batch_5L.append(torch.mean(llama2_batch_outputs.hidden_states[layer], axis=1))    
        llama2_reps_batch_5L = torch.stack(llama2_reps_batch_5L, axis=1)
    #print(llama2_reps_batch_5L)
    return llama2_reps_batch_5L, bert_reps, roberta_reps
           
llama2_reps_batch_5L,bert_reps, roberta_reps =extract_representations(model_tokenizer_dict, sents)
print(llama2_reps_batch_5L,bert_reps, roberta_reps)


