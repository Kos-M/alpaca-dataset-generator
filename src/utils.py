import json
import re
from typing import List, Dict, Any
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from config import CONFIG
import docx
import PyPDF2

# Ensure NLTK data is downloaded
try:
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

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
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('
')

def read_file(file_path: str) -> str:
    """
    Reads content from a given file path, supporting .txt, .pdf, and .docx formats.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The extracted text content from the file.

    Raises:
        ValueError: If the file format is not supported.
    """
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.endswith('.pdf'):
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        return "
".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")

def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses a given text string.

    This function performs the following steps:
    1. Removes extra whitespace.
    2. Removes URLs.
    3. Removes special characters.
    4. Truncates the text to a maximum character length defined in `CONFIG['max_chars']`.

    Args:
        text (str): The input text string to be preprocessed.

    Returns:
        str: The cleaned and preprocessed text string.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text[:CONFIG['max_chars']]

def extract_keywords(text: str, n: int = CONFIG['keyword_count']]) -> List[str]:
    """
    Extracts the most frequent keywords from a given text.

    This function tokenizes the text, removes stopwords, and then identifies
    the `n` most frequent words as keywords.

    Args:
        text (str): The input text from which to extract keywords.
        n (int): The number of keywords to extract (defaults to `CONFIG['keyword_count']`).

    Returns:
        List[str]: A list of extracted keywords.
    """
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stopwords.words('english')]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top n
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    return [word for word, count in sorted_words[:n]]

def calculate_similarity(text1: str, text2: str, model) -> float:
    """
    Calculates the cosine similarity between two text strings using a sentence embedding model.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.
        model: The sentence embedding model (e.g., SentenceTransformer) to generate embeddings.

    Returns:
        float: The cosine similarity score between the embeddings of the two texts.
    """
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def is_valid_output(instruction_type: str, output: str, input_text: str, sentence_model) -> bool:
    """
    Validates a generated output based on its instruction type and content.

    This function applies various validation rules:
    - Checks for minimum word and character counts.
    - Ensures output is not identical to input.
    - For specific instruction types (e.g., 'paraphrase', 'question'), it performs
      additional checks like similarity to input or presence of question marks.

    Args:
        instruction_type (str): The type of instruction that generated the output.
        output (str): The generated text output.
        input_text (str): The original input text used for generation.
        sentence_model: The sentence embedding model for similarity calculations.

    Returns:
        bool: True if the output is considered valid, False otherwise.
    """
    if not output or len(word_tokenize(output)) < CONFIG['min_output_words'] or len(output) < CONFIG['min_output_chars']:
        return False

    if output.strip().lower() == input_text.strip().lower():
        return False

    if instruction_type == "paraphrase":
        similarity = calculate_similarity(output, input_text, sentence_model)
        if similarity > CONFIG['min_similarity_threshold']:
            return True
        return False

    if instruction_type == "question":
        if not output.strip().endswith('?'):
            return False

    if instruction_type == "sentiment":
        if not any(keyword in output.lower() for keyword in ['positive', 'negative', 'neutral']):
            return False

    return True
