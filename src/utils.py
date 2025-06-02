import json
import torch
import os
import numpy as np
from typing import List, Dict, Any, Union
from docx import Document
import PyPDF2
import nltk
from nltk.corpus import stopwords
from collections import Counter
from transformers import PreTrainedTokenizer, PreTrainedModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity

# Import CONFIG if it's defined in a separate file
from config import CONFIG

sentence_model = SentenceTransformer(CONFIG['models']['sentence'])
client = InferenceClient(token=CONFIG['hf_api_token'])

nltk.download('stopwords', quiet=True)

def read_text_file(file_path: str) -> str:
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

def read_pdf_file(file_path: str) -> str:
    """Read content from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Join pages with double newline to preserve paragraph breaks
            return '\n\n'.join([page.extract_text().strip() for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return ""

def read_docx_file(file_path: str) -> str:
    """Read content from a DOCX file."""
    try:
        doc = Document(file_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return ""

def read_file(file_path: str) -> str:
    """Read content from a file based on its extension."""
    _, ext = os.path.splitext(file_path.lower())
    if ext == '.txt':
        return read_text_file(file_path)
    elif ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def preprocess_text(text: str, max_chars: int = CONFIG['max_chars']) -> str:
    """
    Preprocess the input text by:
    1. Removing extra whitespace while preserving paragraph breaks
    2. Truncating to prevent exceeding model token limits
    
    Args:
        text: The input text to preprocess
        max_chars: Maximum number of characters to keep (approximate token limit)
    
    Returns:
        Preprocessed and potentially truncated text
    """
    # Truncate the text if it's too long
    if len(text) > max_chars:
        text = text[:max_chars]
    
    # Split into paragraphs, clean each paragraph, then join with double newline
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    # Replace any hyphens that split words across lines
    cleaned_paragraphs = [' '.join(p.split()).replace('-\n', '').replace(' - ', ' ') for p in paragraphs]
    # Remove non-alphanumeric characters
    cleaned_paragraphs = [''.join(char for char in p if char.isalnum() or char.isspace()) for p in cleaned_paragraphs]
    return '\n\n'.join(cleaned_paragraphs)

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text: str, n: int = 5) -> List[str]:
    """Extract the most common keywords from the text using TF-IDF."""
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    vectorizer.fit([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_matrix = vectorizer.transform([text])
    
    # Get word scores
    word_scores = [(feature_names[col], tfidf_matrix[0, col]) for col in tfidf_matrix.indices]
    
    # Sort by score
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)
    
    # Return top n keywords
    return [word for word, score in sorted_words[:n]]

def generate_gpt2_output(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt: Union[str, List[str]],
    device: torch.device,
    max_length: int = CONFIG['gpt2_output_max_length']
) -> Union[str, List[str]]:
    """Generate output using a GPT-2 model, supporting batch inference."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_single_input = isinstance(prompt, str)
    prompts = [prompt] if is_single_input else prompt

    inputs = tokenizer(prompts, return_tensors="np", padding=True, truncation=True, max_length=512)
    input_ids = torch.from_numpy(inputs.input_ids).to(device)
    attention_mask = torch.from_numpy(inputs.attention_mask).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    
    # Decode outputs. If batched, outputs will be (batch_size * num_return_sequences, sequence_length)
    # We assume num_return_sequences=1 for simplicity here.
    generated_texts = []
    for i in range(len(prompts)):
        # Find the start of the generated text by looking for the prompt's length
        # This assumes that the generated output directly follows the input prompt tokens
        # For batched generation, input_ids.shape[1] might be the max length of the batch inputs
        # A more robust way would be to track original input lengths or use `skip_special_tokens=True`
        # and then remove the prompt from the decoded output.
        decoded_output = tokenizer.decode(outputs[i], skip_special_tokens=True)
        
        # Remove the original prompt from the generated text
        # This is a heuristic and might need refinement based on actual model behavior
        if decoded_output.startswith(prompts[i]):
            generated_text = decoded_output[len(prompts[i]):].strip()
        else:
            generated_text = decoded_output.strip() # Fallback if prompt removal is tricky

        generated_texts.append(generated_text)

    return generated_texts[0] if is_single_input else generated_texts

def generate_t5_output(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prefix: Union[str, List[str]],
    input_text: Union[str, List[str]],
    device: torch.device,
    max_length: int = CONFIG['t5_output_max_length']
) -> Union[str, List[str]]:
    """Generate output using a T5 model, supporting batch inference."""
    is_single_input = isinstance(input_text, str)
    
    if is_single_input:
        prefixes = [prefix]
        input_texts = [input_text]
    else:
        prefixes = prefix if isinstance(prefix, list) else [prefix] * len(input_text)
        input_texts = input_text

    # Prepare inputs for tokenizer
    model_inputs = [f"{p}: {t}" for p, t in zip(prefixes, input_texts)]

    inputs = tokenizer(model_inputs, return_tensors="np", padding=True, truncation=True, max_length=512)
    input_ids = torch.from_numpy(inputs.input_ids).to(device)
    attention_mask = torch.from_numpy(inputs.attention_mask).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return generated_texts[0] if is_single_input else generated_texts

def is_valid_output(
    instruction_type: str,
    output: str,
    input_text: str,
    sentence_model: SentenceTransformer
) -> bool:
    """Validate the generated output based on instruction type and similarity to input."""
    # Check for minimum length
    if len(output.strip()) < 10:
        return False

    # Check for URLs
    if "http://" in output or "https://" in output:
        return False

    if len(output.split()) < CONFIG['min_output_words'] or len(output) < CONFIG['min_output_chars']:
        return False

    input_embedding = sentence_model.encode(input_text)
    output_embedding = sentence_model.encode(output)

    similarity = cosine_similarity(np.array(input_embedding).reshape(1, -1), np.array(output_embedding).reshape(1, -1))[0][0]

    if similarity < CONFIG['min_similarity_threshold']:
        return False

    if instruction_type == "summarize" and (len(output.split()) > CONFIG['max_summarize_words'] or len(output.split()) < CONFIG['min_output_words']):
        return False
    if instruction_type == "keyword" and not (3 <= len(output.split(',')) <= 5):
        return False
    if instruction_type == "title" and (len(output.split()) > 10 or len(output.split()) < 3):
        return False
    if instruction_type == "sentiment" and not any(word in output.lower() for word in ['positive', 'negative', 'neutral']):
        return False
    if instruction_type == "question" and not output.endswith('?'):
        return False
    
    if instruction_type == "concept_explanation":
        keywords = extract_keywords(input_text)
        if not any(keyword.lower() in output.lower() for keyword in keywords):
            return False

    # Check for repeated phrases or sentences
    sentences = nltk.sent_tokenize(output)
    if len(sentences) > 1:
        sentence_counts = Counter(sentences)
        if any(count > 1 for count in sentence_counts.values()):
            return False

    return True

def save_to_jsonl(data: List[Dict[str, Any]], output_file: str):
    """Save data to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
