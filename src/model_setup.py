import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from config import CONFIG
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def setup_models():
    models = {}
    # GPT-2
    models["gpt2_tokenizer"] = GPT2Tokenizer.from_pretrained(CONFIG['models']['gpt2'], padding_side='left')
    models["gpt2_model"] = GPT2LMHeadModel.from_pretrained(CONFIG['models']['gpt2']).to(CONFIG['device'])
    models["gpt2_model"].config.pad_token_id = models["gpt2_model"].config.eos_token_id

    # T5
    models["t5_tokenizer"] = T5Tokenizer.from_pretrained(CONFIG['models']['t5'], legacy=False)
    models["t5_model"] = T5ForConditionalGeneration.from_pretrained(CONFIG['models']['t5']).to(CONFIG['device'])

    # Sentiment analysis with truncation
    models["sentiment_pipeline"] = pipeline(
        "sentiment-analysis",
        model=CONFIG['models']['sentiment'],
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=CONFIG['tokenizer_max_length']
    )

    # Sentence Transformer
    models["sentence_model"] = SentenceTransformer(CONFIG['models']['sentence']).to(CONFIG['device'])

    return models
