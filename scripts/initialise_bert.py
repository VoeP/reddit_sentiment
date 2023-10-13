"""Single purpose script to load a Bert model and the emotions model
Bert requires additional downloads to work
Running this during container image building will ensure that the model requirements are pre-downloaded"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# Generic function for loading a huggingface model
def load_huggingface_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Generic function for getting tokenizer and model from huggingface
def load_huggingface_tokenizer_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    return tokenizer, model


pipeline_bert = load_huggingface_model("nlptown/bert-base-multilingual-uncased-sentiment")
emotions_tokeniser, emotions_model = load_huggingface_tokenizer_model("cardiffnlp/twitter-roberta-base-emotion")
