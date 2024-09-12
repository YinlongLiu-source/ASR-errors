import os
import configparser
import logging
import shutil
import sys
import gc
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union,Any
import torch
import pandas as pd
import soundfile as sf
import numpy as np
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration,WhisperProcessor,WhisperFeatureExtractor,WhisperTokenizer
from evaluate import load

os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def parge_args():
    """
    add argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str,
        help="""The name of huggingface model card""")
    parser.add_argument("--experiment_id", type=str, help="The experiment id of evaluation")
    parser.set_defaults(feature=True)
    return parser.parse_args()


def load_model(model_name):
    """
    load the model and corresponding processor with model card name

    :param model_name: the name of model card from huggingface
    :type model_name: str
    :return: model and processor
    :rtype: transformers.AutoModelForCTC, transformers.Wav2Vec2Processor
    """
    model_path=f"/mnt/hd/data_lyl/projects/huggingface_model/{model_name}"
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    processor = WhisperProcessor.from_pretrained(model_path,language="english", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path,language="english", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained(model_path,language="english", task="transcribe")
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    return model, processor,feature_extractor,tokenizer


def speech_file_to_array_fn(batch):
    """
    batchfy and map the wav file to arrays

    :param batch: _description_
    :type batch: _type_
    :return: _description_
    :rtype: _type_
    """
    # print(batch["locs"])
    # print(type(batch["locs"]))
    # speech_array, sampling_rate = sf.read(batch["locs"])
    try:
        speech_array, sampling_rate = sf.read(batch["locs"])
    except Exception as e:
        print(f"Error reading file {batch['locs']}: {e}")
        raise
    # logging.info(f"Processing {batch['locs']}")
    batch["speech"] = speech_array
    # logging.info(f"Speech_array: {speech_array}")
    batch["sampling_rate"] = sampling_rate
    batch["locs"] = batch["locs"]
    return batch


def prepare_dataset(batch):

    # compute log-Mel input features from input audio array  
    # print('batch:',batch)

    inputs = feature_extractor(batch["speech"], sampling_rate=16000,return_tensors="pt")
    batch["input_features"] = inputs.input_features[0]
    # logging.info(f"input_features: {batch['input_features']}")
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["trans"],padding=True).input_ids

    # logging.info(f"labels: {batch['labels']}")
    return batch


    # inputs = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0],padding=True)
    # batch["input_values"] = inputs.input_values
    # # with processor.as_target_processor():
    # #     batch["labels"] = processor(batch["trans"], padding=True).input_ids
    # batch["labels"] = processor(text=batch["trans"], padding=True).input_ids
    # return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id #.tokenizer

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4), "cer": round(cer, 4)}


if __name__ == "__main__":
    start_time = datetime.now()
    args = parge_args()
    wer_metric = load("/mnt/hd/data_lyl/projects/utils/evaluate-0.1.2/metrics/wer", experiment_id=f'{args.experiment_id}')
    cer_metric = load("/mnt/hd/data_lyl/projects/utils/evaluate-0.1.2/metrics/cer", experiment_id=f'{args.experiment_id}')
    config = configparser.ConfigParser()
    config.read("config.ini") #include the related path

    output_log = f"../logs/ft-{args.model_name}"
    model, processor,feature_extractor,tokenizer = load_model(args.model_name)
    print('model:',model)
    # print('processor:',processor)
    log = open(output_log, "w")
    sys.stdout = log
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        filemode="a", level=logging.INFO,
        filename=output_log)

    wls_dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(
            config["PATH"]["train_audio_dir"], "train.csv"),
            "test": os.path.join(
            config["PATH"]["test_audio_dir"], "test.csv")})
    print(wls_dataset)
    sys.stdout.write(f"Fine-tuning {args.model_name}...\n")
    # model.freeze_feature_encoder()
    wls_dataset = wls_dataset.map(
        speech_file_to_array_fn, num_proc=32)
    # print(wls_dataset['test'][0])
    # from datasets import Audio
    # wls_dataset = wls_dataset.cast_column("speech", Audio(sampling_rate=16000))

    wls_dataset = wls_dataset.map(prepare_dataset,remove_columns=['locs', 'trans', 'speech', 'sampling_rate']) #,remove_columns=['locs', 'trans', 'speech', 'sampling_rate']
    print(wls_dataset)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,)
    print(wls_dataset)
    # data_collator = DataCollatorCTCWithPadding(
    #     processor=processor, padding=True)
    # fine-tuning process
    
    epochs=20
    batch=8
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"../ft-models/asr/{args.model_name}",
        group_by_length=True,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        gradient_accumulation_steps=1,
        # evaluation_strategy="epoch",
        # logging_strategy="epoch",
        # save_strategy="epoch",
        # freeze_feature_encoder="True",
        evaluation_strategy="steps",
        save_steps=80,
        eval_steps=80,
        logging_steps=20,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        predict_with_generate=True,
        generation_max_length=225,
        do_train=True,
        do_eval=False,
        save_total_limit=2,
        logging_dir='../logs',
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=42,
        data_seed=42,
        report_to="none",
        load_best_model_at_end=True)
    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        train_dataset=wls_dataset["train"],
        eval_dataset=wls_dataset["test"])

    trainer.train()
    # save the fine-tuned model
    model.save_pretrained(
        os.path.join(
            config["PATH"]["PrefixModel"], f"{args.model_name}")
    )
    # shutil.rmtree("../ft-models/asr/")
    # # remove things to free memory
    # del model, processor, trainer
    gc.collect()
    sys.stdout.write(f"Running time: {datetime.now() - start_time}\n")
    log.close()