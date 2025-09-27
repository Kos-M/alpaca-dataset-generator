import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import read_file, preprocess_text
from config import CONFIG

def load_input_data(input_folder: str) -> List[str]:
    """
    Loads and preprocesses text data from various file formats within a specified folder.

    This function recursively walks through the `input_folder`, identifies supported file types
    (.txt, .pdf, .docx), and processes each file in parallel using a thread pool.
    For each file, it extracts text content, splits it into paragraphs, and preprocesses
    each paragraph before adding it to a collective list of texts.

    Args:
        input_folder (str): The path to the folder containing the input text files.

    Returns:
        List[str]: A list of preprocessed text paragraphs extracted from all input files.
                   Returns an empty list if no valid files are found or an error occurs.
    """
    texts = []
    total_files = sum(len(files) for _, _, files in os.walk(input_folder))
    print(f"Found {total_files} files to process")
    
    def process_file(file_path):
        """Processes a single file, extracts text, and preprocesses paragraphs."""
        try:
            print(f"Processing file: {file_path}")
            text = read_file(file_path)
            paragraphs = [preprocess_text(para) for para in text.split('

') if para.strip()]
            print(f"Extracted {len(paragraphs)} paragraphs from {file_path}")
            return paragraphs
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []

    with tqdm(total=total_files, desc="Loading input files", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
            futures = []
            for root, _, files in os.walk(input_folder):
                for file in files:
                    if file.lower().endswith(('.txt', '.pdf', '.docx')):
                        file_path = os.path.join(root, file)
                        futures.append(executor.submit(process_file, file_path))
            
            for future in as_completed(futures):
                result = future.result()
                texts.extend(result)
                print(f"Added {len(result)} paragraphs to texts")
                pbar.update(1)

    print(f"Total paragraphs loaded: {len(texts)}")
    return texts
