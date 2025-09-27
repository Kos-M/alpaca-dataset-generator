import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
from config import CONFIG
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def setup_models():
    """
    Initializes and configures all necessary language models and tokenizers.

    This function loads pre-trained models for GPT-2, T5, sentiment analysis, and sentence embeddings
    based on the configurations specified in `CONFIG['models']`. It also sets up tokenizers
    and moves models to the appropriate device (GPU if available, otherwise CPU).

    Returns:
        dict: A dictionary containing the initialized models and tokenizers:
              - "gpt2_tokenizer": GPT-2 tokenizer.
              - "gpt2_model": GPT-2 language model.
              - "t5_tokenizer": T5 tokenizer.
              - "t5_model": T5 for conditional generation model.
              - "sentiment_pipeline": Hugging Face sentiment analysis pipeline.
              - "sentence_model": Sentence Transformer model for embeddings.
    """
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
