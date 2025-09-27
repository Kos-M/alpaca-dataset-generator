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
    """
    Reads the entire content of a plain text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the file as a single string. Returns an empty string
             if an IOError occurs during reading.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading text file {file_path}: {e}")
        return ""

def read_pdf_file(file_path: str) -> str:
    """
    Reads the text content from a PDF file.

    Each page's extracted text is joined by a double newline to help preserve
    paragraph separation.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text content from the PDF. Returns an empty string
             if an error occurs during reading.
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Join pages with double newline to preserve paragraph breaks
            return '

'.join([page.extract_text().strip() for page in reader.pages])
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return ""

def read_docx_file(file_path: str) -> str:
    """
    Reads the text content from a DOCX file.

    Paragraphs are joined by a single space.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text content from the DOCX. Returns an empty string
             if an error occurs during reading.
    """
    try:
        doc = Document(file_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file {file_path}: {e}")
        return ""

def read_file(file_path: str) -> str:
    """
    Reads content from a file based on its extension.

    This function acts as a dispatcher, calling the appropriate file reading
    utility (`read_text_file`, `read_pdf_file`, `read_docx_file`) based on
    the file's extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text content from the file.

    Raises:
        ValueError: If the file type is not supported (.txt, .pdf, .docx).
    """
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
    Preprocesses the input text by cleaning and truncating it.

    This function performs the following steps:
    1. Truncates the text to a maximum character length to prevent exceeding model token limits.
    2. Splits the text into paragraphs, cleans each paragraph by removing extra whitespace,
       replacing hyphenated line breaks, and removing non-alphanumeric characters.
    3. Joins the cleaned paragraphs back with double newlines.

    Args:
        text (str): The input text to preprocess.
        max_chars (int): The maximum number of characters to keep (approximate token limit).
                         Defaults to `CONFIG['max_chars']`.

    Returns:
        str: The preprocessed and potentially truncated text.
    """
    # Truncate the text if it's too long
    if len(text) > max_chars:
        text = text[:max_chars]
    
    # Split into paragraphs, clean each paragraph, then join with double newline
    paragraphs = [p.strip() for p in text.split('

') if p.strip()]
    # Replace any hyphens that split words across lines
    cleaned_paragraphs = [' '.join(p.split()).replace('-
', '').replace(' - ', ' ') for p in paragraphs]
    # Remove non-alphanumeric characters
    cleaned_paragraphs = [''.join(char for char in p if char.isalnum() or char.isspace()) for p in cleaned_paragraphs]
    return '

'.join(cleaned_paragraphs)

from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text: str, n: int = 5) -> List[str]:
    """
    Extracts the most common keywords from the text using TF-IDF.

    This function tokenizes the text, removes stopwords, calculates TF-IDF scores
    for each word, and returns the top `n` words with the highest scores as keywords.

    Args:
        text (str): The input text from which to extract keywords.
        n (int): The number of keywords to extract. Defaults to 5.

    Returns:
        List[str]: A list of extracted keywords.
    """
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
    """
    Generates text output using a GPT-2 model.

    This function supports both single and batch inference. It tokenizes the input
    prompt(s), generates text using the GPT-2 model, and then decodes the generated
    tokens back into human-readable text. It attempts to remove the original prompt
    from the generated output.

    Args:
        tokenizer (PreTrainedTokenizer): The GPT-2 tokenizer.
        model (PreTrainedModel): The GPT-2 language model.
        prompt (Union[str, List[str]]): The input prompt(s) for text generation.
                                        Can be a single string or a list of strings.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run the model on.
        max_length (int): The maximum length of the generated output sequence (excluding prompt).
                          Defaults to `CONFIG['gpt2_output_max_length']`.

    Returns:
        Union[str, List[str]]: The generated text output(s). Returns a single string
                               if a single prompt was provided, otherwise a list of strings.
    """
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
    """
    Generates text output using a T5 model for conditional generation tasks.

    This function supports both single and batch inference. It constructs model inputs
    by combining a prefix (e.g., "summarize:") with the input text, tokenizes them,
    generates text using the T5 model, and then decodes the generated tokens.

    Args:
        tokenizer (PreTrainedTokenizer): The T5 tokenizer.
        model (PreTrainedModel): The T5 conditional generation model.
        prefix (Union[str, List[str]]): The prefix(es) to prepend to the input text.
                                        Can be a single string or a list of strings.
        input_text (Union[str, List[str]]): The input text(s) for generation.
                                            Can be a single string or a list of strings.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') to run the model on.
        max_length (int): The maximum length of the generated output sequence.
                          Defaults to `CONFIG['t5_output_max_length']`.

    Returns:
        Union[str, List[str]]: The generated text output(s). Returns a single string
                               if a single input was provided, otherwise a list of strings.
    """
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
    """
    Validates a generated output based on its instruction type and content.

    This function applies various validation rules:
    - Checks for minimum length (characters and words).
    - Checks for the presence of URLs.
    - Calculates cosine similarity between the output and input to ensure relevance.
    - Applies specific validation rules based on the `instruction_type` (e.g., summarization
      length, keyword count, title length, sentiment keywords, question mark presence,
      concept explanation keyword presence).
    - Checks for repeated phrases or sentences within the output.

    Args:
        instruction_type (str): The type of instruction that generated the output.
        output (str): The generated text output.
        input_text (str): The original input text used for generation.
        sentence_model (SentenceTransformer): The sentence embedding model for similarity calculations.

    Returns:
        bool: True if the output is considered valid, False otherwise.
    """
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
    """
    Saves a list of dictionaries to a JSONL (JSON Lines) file.

    Each dictionary in the list is serialized to a JSON string and written
    as a new line in the specified output file.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries to be saved.
        output_file (str): The path to the output JSONL file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('
')
