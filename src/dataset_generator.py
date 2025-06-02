import random
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from utils import generate_gpt2_output, generate_t5_output, extract_keywords, preprocess_text, is_valid_output
from config import CONFIG
from model_setup import setup_models
import re

class TextDataset(Dataset):
    def __init__(self, texts, instructions):
        # Preprocess texts when loading to ensure they're within token limits
        self.texts = [preprocess_text(text) for text in texts]
        self.instructions = instructions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        instruction_type, instruction, prompt_template = random.choice(self.instructions)
        return text, instruction_type, instruction

def generate_dataset(input_texts: List[str], models: Dict) -> List[Dict[str, Any]]:
    dataset = TextDataset(input_texts, CONFIG['instruction_types'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['max_workers'])

    examples = []

    with tqdm(total=CONFIG['num_examples'], desc="Generating examples", unit="example") as pbar:
        try:
            for batch in dataloader:
                texts, instruction_types, instructions = batch
                batch_examples = generate_batch(models, texts, instruction_types, instructions)
                examples.extend(batch_examples)
                pbar.update(len(batch_examples))
                if len(examples) >= CONFIG['num_examples']:
                    break
        except Exception as e:
            print(f"Error during dataset generation: {e}")

    return examples[:CONFIG['num_examples']]

def generate_batch(models: Dict, texts: List[str], instruction_types: List[str], instructions: List[str]) -> List[Dict[str, Any]]:
    batch_examples_data = []
    t5_tasks = []
    gpt2_tasks = []
    sentiment_tasks = []
    keyword_tasks = []

    # Collect tasks for batching
    for i, (text, instruction_type, instruction) in enumerate(zip(texts, instruction_types, instructions)):
        # Summarize the input text using T5
        if models.get("t5_tokenizer") and models.get("t5_model"):
            summarized_text = generate_t5_output(models["t5_tokenizer"], models["t5_model"], "summarize", text, CONFIG['device'], max_length=CONFIG['t5_prompt_max_length'])
        else:
            summarized_text = text  # If T5 is not available, use the original text

        batch_examples_data.append({
            "instruction": instruction,
            "input": summarized_text,
            "instruction_type": instruction_type,
            "output": None,  # Placeholder for output
            "original_index": i  # Keep track of original order
        })

        if instruction_type in ["summarize", "paraphrase"]:
            t5_tasks.append({"text": summarized_text, "type": instruction_type, "index": i})
        elif instruction_type == "keyword":
            keyword_tasks.append({"text": summarized_text, "index": i})
        elif instruction_type == "sentiment":
            sentiment_tasks.append({"text": summarized_text, "index": i})
        else: # For other GPT-2 tasks (title, question, paraphrase, etc.)
            prompt_template = next(item[2] for item in CONFIG['instruction_types'] if item[0] == instruction_type)
            prompt = prompt_template.format(text=summarized_text, related_concept=extract_keywords(summarized_text, n=1)[0] if "concept_relation" in instruction_type else summarized_text)
            gpt2_tasks.append({"prompt": prompt, "index": i})

    # Process T5 tasks in batch
    if t5_tasks and models.get("t5_tokenizer") and models.get("t5_model"):
        t5_texts = [task["text"] for task in t5_tasks]
        t5_types = [task["type"] for task in t5_tasks]
        t5_outputs = generate_t5_output(models["t5_tokenizer"], models["t5_model"], t5_types, t5_texts, CONFIG['device'])
        for j, output in enumerate(t5_outputs):
            original_index = t5_tasks[j]["index"]
            batch_examples_data[original_index]["output"] = output

    # Process GPT-2 tasks in batch
    if gpt2_tasks:
        gpt2_prompts = [task["prompt"] for task in gpt2_tasks]
        gpt2_outputs = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], gpt2_prompts, CONFIG['device'])
        for j, output in enumerate(gpt2_outputs):
            original_index = gpt2_tasks[j]["index"]
            batch_examples_data[original_index]["output"] = output

    # Process Keyword tasks in batch
    if keyword_tasks:
        for task in keyword_tasks:
            keywords = extract_keywords(task["text"])
            output = ", ".join(keywords)
            batch_examples_data[task["index"]]["output"] = output

    # Process Sentiment tasks in batch
    if sentiment_tasks and models.get("sentiment_pipeline"):
        sentiment_texts = [task["text"] for task in sentiment_tasks]
        # Add truncation to handle long texts
        truncated_sentiment_texts = [text[:CONFIG['sentiment_truncation_length']] for text in sentiment_texts]
        sentiments = models["sentiment_pipeline"](truncated_sentiment_texts)
        
        # Convert sentiment analysis results to numpy arrays
        sentiments = [{"label": s["label"], "score": s["score"]} for s in sentiments]

        # Prepare prompts for GPT-2 explanation of sentiment
        gpt2_sentiment_explanation_tasks = []
        for j, sentiment in enumerate(sentiments):
            original_index = sentiment_tasks[j]["index"]
            prompt = f"Explain why the sentiment is {sentiment['label']}: "
            gpt2_sentiment_explanation_tasks.append({"prompt": prompt, "original_index": original_index, "sentiment_label": sentiment['label']})

        if gpt2_sentiment_explanation_tasks:
            gpt2_explanation_prompts = [task["prompt"] for task in gpt2_sentiment_explanation_tasks]
            gpt2_explanations = generate_gpt2_output(models["gpt2_tokenizer"], models["gpt2_model"], gpt2_explanation_prompts, CONFIG['device'], max_length=CONFIG['gpt2_max_length'])
            for j, explanation in enumerate(gpt2_explanations):
                original_index = gpt2_sentiment_explanation_tasks[j]["original_index"]
                sentiment_label = gpt2_sentiment_explanation_tasks[j]["sentiment_label"]
                output = f"{sentiment_label.capitalize()}. {explanation}"
                batch_examples_data[original_index]["output"] = output

    # Filter out examples that might not have been processed (shouldn't happen with this logic)
    final_batch_examples = [example for example in batch_examples_data if example["output"] is not None and is_valid_output(batch_examples_data[example["original_index"]]["instruction_type"], batch_examples_data[example["original_index"]]["output"], batch_examples_data[example["original_index"]]["input"], models["sentence_model"])]

    return final_batch_examples