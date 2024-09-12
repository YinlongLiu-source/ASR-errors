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


os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
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
    model = AutoModelForCTC.from_pretrained(f"/mnt/hd/data_lyl/projects/huggingface_model/{model_name}")
    processor = Wav2Vec2Processor.from_pretrained(f"/mnt/hd/data_lyl/projects/huggingface_model/{model_name}")
    return model, processor


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
    """
    batchfy and encode the transcripts to label ids

    :param batch: _description_
    :type batch: _type_
    :return: _description_
    :rtype: _type_
    """
    inputs = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0],padding=True)
    batch["input_values"] = inputs.input_values
    # with processor.as_target_processor():
    #     batch["labels"] = processor(batch["trans"], padding=True).input_ids
    batch["labels"] = processor(text=batch["trans"], padding=True).input_ids
    return batch


def compute_metrics(pred):
    """
    batchfy and compute the WER metrics

    :param pred: _description_
    :type pred: _type_
    :return: _description_
    :rtype: _type_
    """
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    # logging.info(f"Prediction: {pred_str}")
    # logging.info(f"Reference: {label_str}")
    return {"wer": round(wer, 4), "cer": round(cer, 4)}

if __name__ == "__main__":
    start_time = datetime.now()
    args = parge_args()
    wer_metric = load("./evaluate-0.1.2/metrics/wer", experiment_id=f'{args.experiment_id}')
    cer_metric = load("./evaluate-0.1.2/metrics/cer", experiment_id=f'{args.experiment_id}')
    config = configparser.ConfigParser()
    config.read("config.ini") #include the related path
    batch = 4
    epochs = 20
    output_log = f"../logs/ft-{args.model_name}"
    model, processor = load_model(args.model_name)
    log = open(output_log, "w")
    sys.stdout = log
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        filemode="a", level=logging.INFO,
        filename=output_log)
    # use 00 and 01 as training set
    
    # wls_00 = pd.read_csv(
    #     os.path.join(
    #         config["PATH"]["PrefixManifest"], "wls_00_trans.csv")
    # )
    # wls_01 = pd.read_csv(
    #     os.path.join(
    #         config["PATH"]["PrefixManifest"], "wls_01_trans.csv")
    # )
    # wls_train = pd.concat([wls_00, wls_01])
    
    # wls_train.to_csv(
    #     os.path.join(
    #         config["PATH"]["PrefixManifest"], "wls_train.csv"),index=False)
    wls_dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(
            config["PATH"]["train_audio_dir"], "train.csv"),
            "test": os.path.join(
            config["PATH"]["test_audio_dir"], "test.csv")})
    sys.stdout.write(f"Fine-tuning {args.model_name}...\n")
    model.freeze_feature_encoder()
    wls_dataset = wls_dataset.map(
        speech_file_to_array_fn, num_proc=32)
    wls_dataset = wls_dataset.map(
        prepare_dataset, batch_size=batch,
        num_proc=32, batched=True,remove_columns=['locs', 'trans', 'speech', 'sampling_rate'])
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding=True)
    # fine-tuning process
    training_args = TrainingArguments(
        output_dir=f"../ft-models/asr/{args.model_name}",
        group_by_length=True,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        gradient_accumulation_steps=2,
        # evaluation_strategy="epoch",
        # logging_strategy="epoch",
        # save_strategy="epoch",
        evaluation_strategy="steps",
        save_steps=80,
        eval_steps=80,
        logging_steps=20,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=100,
        fp16=True,
        gradient_checkpointing=True,
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
    trainer = Trainer(
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