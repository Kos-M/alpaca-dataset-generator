import torch

CONFIG = {
    'input_folder': 'path/to/input/folder',
    'output_file': 'path/to/output.jsonl',
    'validated_output_file': 'path/to/validated_output.jsonl',
    'num_examples': 5,
    'batch_size': 5,
    'max_workers': 2,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'models': {
        'gpt2': 'gpt2-large',
        't5': 't5-large',
        'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
        'sentence': 'all-MiniLM-L6-v2'
    },
    # Text processing parameters
    'keyword_count': 5,
    
    # Generation parameters
    'gpt2_max_length': 80,
    'no_repeat_ngram_size': 2,
    't5_prompt_max_length': 250,
    'tokenizer_max_length': 512,
    'top_k': 45,
    'top_p': 0.90,

    'sentiment_truncation_length': 1000,

    # Text processing parameters
    'max_chars': 500,
    
    # Generation parameters
    'gpt2_output_max_length': 80,
    't5_output_max_length': 150,

    # Validation parameters
    'min_word_count': 10,
    'min_output_length': 50,
    'min_similarity_threshold': 0.3,
    'min_explanation_word_count': 10,
    'min_output_words': 2,
    'min_output_chars': 5,
    'max_summarize_words': 50,
    'min_learning_path_stepspath_steps': 1,
    'min_quiz_options': 2,
    'difficulty_keywords': ['easy', 'medium', 'difficult', 'challenging'],
    'use_huggingface_embeddings': True,
    'hf_api_token': 'yourTokenHere',

    # Instruction types for dataset generation
    'instruction_types': [
        ("concept_explanation", "Explain the following concept in simple terms, focusing on its key aspects and providing a clear and concise definition:", "Concept: {text}"),
        ("generate_question", "Generate a thought-provoking question about the following concept:", "Concept: {text}\\n\\nOutput:"),
        ("provide_example", "Provide a real-world example that illustrates the following concept:", "Concept: {text}\\n\\nExplanation:"),
        ("keyword", "Extract 3-5 main keywords or key phrases from the following text:", "Text: {text}\\n\\nOutput:"),
        ("title", "Generate a short, engaging title for the following text:", "Text: {text}\\n\\nOutput:"),
        ("sentiment", "Analyze the sentiment of the following text. Classify it as positive, negative, or neutral, and briefly explain your reasoning:", "Text: {text}\\n\\nOutput:"),
        ("question", "Generate a thought-provoking question based on the main idea of the following text:", "Text: {text}\\n\\nOutput:"),
        ("paraphrase", "Rewrite the following text in your own words, maintaining its core meaning:", "Text: {text}\\n\\nOutput:")

    ]
}
