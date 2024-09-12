# ASR-errors

The resource of paper "Can Automated Speech Recognition Errors Provide Valuable Clues for Alzheimer’s Disease Detection?"

## Please set up the environment first：

```
pip install -r requirements.txt
```

## Data

To obtain the data used for fine-tuning the ASR models (WLS, Lu, Kempler) and for AD detection (ADReSS), please visit [DementiaBank](https://dementia.talkbank.org/) , as we do not have the authorization to make them publicly available.

## Methods

### **Fine tune ASR models:**

* **fine_tune_asr_wav2vec.py :** fine tune Wav2Vec 2.0, HuBERT, and WavLM models.
* **fine_tune_asr_whisper.py :** fine tune Whisper model.
* **eval_wav2vec.py:** evaluate Wav2Vec 2.0, HuBERT, and WavLM models.
* **eval_whisper.py:** evaluate Whisper model.

### AD detection

#### Fine tune LLMs

* **fine_tune_llm_llama3.1.py:** fine tune LLMs (Llama3.1 can be replaced with Qwen2, Mistral, and Glm).

#### **Fusion of LLMs with PLMs** 

* **extract_embedding.py** : extract the embeddings.
* **MyDataset.py** : store the dataset.
* **DownstreamModel.p**y : construct the model.
* **model_op.py :** optimize the model.
* **main.py:** run experiments.

### **Interpretability Study**

* **Linguistic analysis.ipynb:** conduct linguistic analysis.
* **SHAP analysis.ipynb:** conduct SHAP analysis.
