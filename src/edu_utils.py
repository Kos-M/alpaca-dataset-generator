import json
import torch
import os
from typing import List, Dict, Any
from docx import Document
import PyPDF2
from nltk.corpus import stopwords
from collections import Counter
from config import CONFIG

def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path: str) -> str:
    reader = PyPDF2.PdfReader(file)
    return ' '.join([page.extract_text() for page in reader.pages])

def read_docx_file(file_path: str) -> str:
    doc = Document(file_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def read_file(file_path: str) -> str:
    _, ext = os.path.splitext(file_path.lower())
    if ext == '.txt':
        return read_text_file(file_path)
    elif ext == '.pdf':
        return read_pdf_file(file_path)
    elif ext == '.docx':
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def preprocess_text(text: str) -> str:
    return ' '.join(text.split())

def extract_keywords(text: str, n: int = CONFIG['keyword_count']) -> List[str]:
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in text.split() if word.isalnum()]
    word_freq = Counter(word for word in words if word not in stop_words)
    return [word for word, _ in word_freq.most_common(n)]

def generate_gpt2_output(tokenizer, model, prompt: str, device: torch.device, max_length: int = CONFIG['gpt2_max_length']) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=len(input_ids[0]) + max_length, num_return_sequences=1, no_repeat_ngram_size=CONFIG['no_repeat_ngram_size'], pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

def generate_t5_output(tokenizer, model, prefix: str, input_text: str, device: torch.device, max_length: int = CONFIG['t5_max_length']) -> str:
    input_ids = tokenizer(f"{prefix}: {input_text}", return_tensors="pt", max_length=CONFIG['tokenizer_max_length'], truncation=True).input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True, top_k=CONFIG['top_k'], top_p=CONFIG['top_p'])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def is_valid_output(instruction_type: str, output: str, input_text: str, sentence_model) -> bool:
    if len(output.split()) < CONFIG['min_word_count'] or len(output) < CONFIG['min_output_length']:
        return False

    input_embedding = sentence_model.encode(input_text, convert_to_tensor=True)
    output_embedding = sentence_model.encode(output, convert_to_tensor=True)
    similarity = torch.cosine_similarity(input_embedding, output_embedding, dim=0).item()

    if similarity < CONFIG['min_similarity_threshold']:
        return False

    if instruction_type == "concept_explanation" and len(output.split()) < CONFIG['min_explanation_word_count']:
        return False
    if instruction_type == "generate_question" and not output.endswith('?'):
        return False
    if instruction_type == "provide_example" and "for example" not in output.lower():
        return False
    if instruction_type == "learning_path" and len(output.split(',')) < CONFIG['min_learning_path_steps']:
        return False
    if instruction_type == "misconception" and "however" not in output.lower():
        return False
    if instruction_type == "analogy" and "like" not in output.lower():
        return False
    if instruction_type == "quiz_generation" and output.count(')') < CONFIG['min_quiz_options']:
        return False
    if instruction_type == "concept_relation" and "related" not in output.lower():
        return False
    if instruction_type == "application" and "used" not in output.lower():
        return False
    if instruction_type == "difficulty_assessment" and not any(word in output.lower() for word in CONFIG['difficulty_keywords']):
        return False

    return True

def save_to_jsonl(data: List[Dict[str, Any]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
