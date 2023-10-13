"""Single purpose script to load a Bert model
Bert requires additional downloads to work
Running this during container image building will ensure that the model requirements are pre-downloaded"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# Generic function for loading a huggingface model
def load_huggingface_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return TextClassificationPipeline(model=model, tokenizer=tokenizer)

pipeline_bert = load_huggingface_model("nlptown/bert-base-multilingual-uncased-sentiment")
