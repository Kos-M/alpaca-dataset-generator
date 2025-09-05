# Alpaca-style Dataset Generator

This project generates a high-quality Alpaca-style dataset from input text files, PDFs, and Word documents. It features optimized performance, GPU acceleration, and customizable output.

## Features

- Multi-threaded data loading from various file formats (txt, pdf, docx)
- Batch processing for efficient dataset generation
- GPU acceleration (if available)
- Separate raw and validated output files
- Progress tracking for all major steps
- Customizable configuration

## Project Structure

```
alpaca-dataset-generator/
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── data_loader.py
│   ├── model_setup.py
│   ├── dataset_generator.py
│   ├── validation.py
│   └── utils.py
│
├── data/
│   └── input/
│       ├── file1.txt
│       ├── file2.pdf
│       └── file3.docx
│
├── output/
│   ├── raw_dataset.jsonl
│   └── validated_dataset.jsonl
│
├── requirements.txt
└── README.md
```

## Code Overview

This section provides a brief overview of key functions within the project.

### `config.py`

- `CONFIG`: A dictionary containing all the configuration parameters for the dataset generation process.
    - `input_folder` (str): Path to your input data folder.
    - `output_file` (str): Path for the raw output file.
    - `validated_output_file` (str): Path for the validated output file.
    - `num_examples` (int): Number of examples to generate.
    - `batch_size` (int): Batch size for processing.
    - `max_workers` (int): Number of worker threads for data loading.
    - `device` (torch.device): The device to use for computations (GPU if available, otherwise CPU).
    - `models` (dict): A dictionary specifying the models to be used.
        - `gpt2` (str): The GPT-2 model to use (e.g., 'gpt2-large').
        - `t5` (str): The T5 model to use (e.g., 't5-large').
        - `sentiment` (str): The sentiment analysis model to use (e.g., 'distilbert-base-uncased-finetuned-sst-2-english').
        - `sentence` (str): The sentence embedding model to use (e.g., 'all-MiniLM-L6-v2').
    - `keyword_count` (int): Number of keywords to extract.
    - `gpt2_max_length` (int): Maximum length for GPT-2 generated sequences.
    - `no_repeat_ngram_size` (int): Size of n-grams to avoid repeating in generated text.
    - `t5_prompt_max_length` (int): Maximum length for T5 prompt.
    - `tokenizer_max_length` (int): Maximum length for tokenizer.
    - `top_k` (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
    - `top_p` (float): The cumulative probability for top-p-filtering.
    - `sentiment_truncation_length` (int): Length to truncate text for sentiment analysis.
    - `max_chars` (int): Maximum characters to process from a text.
    - `gpt2_output_max_length` (int): Maximum length for GPT-2 output.
    - `t5_output_max_length` (int): Maximum length for T5 output.
    - `min_word_count` (int): Minimum word count for validation.
    - `min_output_length` (int): Minimum output length for validation.
    - `min_similarity_threshold` (float): Minimum similarity threshold for validation.
    - `min_explanation_word_count` (int): Minimum word count for explanations.
    - `min_output_words` (int): Minimum number of words in the output.
    - `min_output_chars` (int): Minimum number of characters in the output.
    - `max_summarize_words` (int): Maximum number of words for summarization.
    - `min_learning_path_stepspath_steps` (int): Minimum number of steps in a learning path.
    - `min_quiz_options` (int): Minimum number of options for a quiz question.
    - `difficulty_keywords` (list): Keywords to identify difficulty levels.
    - `use_huggingface_embeddings` (bool): Whether to use Hugging Face embeddings.
    - `hf_api_token` (str): Hugging Face API token.
    - `instruction_types` (list): A list of tuples, each defining an instruction type for dataset generation. Each tuple contains:
        - (str): A unique identifier for the instruction type (e.g., "concept_explanation").
        - (str): The instruction prompt to be given to the model.
        - (str): The format string for the input text.

### `main.py`

- `main()`: The main function that orchestrates the dataset generation process. It loads input data, sets up models, generates the dataset, saves the raw dataset, validates the generated examples, and saves the validated dataset.

### `validation.py`

- `validate_dataset(dataset: List[Dict[str, Any]], sentence_model) -> List[Dict[str, Any]]`: Validates the generated dataset based on predefined criteria. It iterates through each example in the dataset and uses `is_valid_output` from `utils.py` to check its validity.

### `data_loader.py`

- `process_file(file_path)`: Processes a single input file (txt, pdf, or docx) and extracts its content.

### `model_setup.py`

- `setup_models()`: Initializes and configures the language model and tokenizer used for dataset generation.

### `utils.py`

- `save_to_jsonl(data, output_file)`: Saves a list of dictionaries to a JSONL (JSON Lines) file.

### `dataset_generator.py`

- `TextDataset(Dataset)`: A custom PyTorch Dataset class for handling text data.
  - `__init__(self, texts, instructions)`: Initializes the dataset with a list of texts and instructions.
  - `__len__(self)`: Returns the total number of texts in the dataset.
  - `__getitem__(self, idx)`: Retrieves a text and a randomly chosen instruction for a given index.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/ekatraone/alpaca-dataset-generator.git
cd alpaca-dataset-generator
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Download NLTK data:

```bash
python -m nltk.downloader punkt stopwords
```

## Configuration

Open `src/config.py` and adjust the settings as needed:

- `input_folder`: Path to your input data folder (default: 'data/input')
- `output_file`: Path for the raw output file (default: 'output/raw_dataset.jsonl')
- `validated_output_file`: Path for the validated output file (default: 'output/validated_dataset.jsonl')
- `num_examples`: Number of examples to generate
- `batch_size`: Batch size for processing
- `max_workers`: Number of worker threads for data loading

## Usage

1. Place your input files (.txt, .pdf, .docx) in the `data/input/` directory.

2. Run the script:

```bash
python src/main.py --num_examples 1000
```

   * `--num_examples`: Number of examples to generate (default: 1000)

3. The script will generate two files in the `output/` directory:
   - `raw_dataset.jsonl`: Contains all generated examples
   - `validated_dataset.jsonl`: Contains only the examples that passed validation

## Customization

- To modify the types of examples generated, edit the `instruction_types` list in `src/config.py`.
- To adjust validation criteria, modify the `is_valid_output` function in `src/utils.py`.

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the `batch_size` in `src/config.py`.
- If the process is too slow, you can try increasing `max_workers` or `batch_size`, but be cautious of memory usage.

## Releases

For information about the latest releases and changes, please refer to the CHANGELOG.md file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
