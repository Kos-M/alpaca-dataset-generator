from data_loader import load_input_data
from model_setup import setup_models
from dataset_generator import generate_dataset
from validation import validate_dataset
from utils import save_to_jsonl
from config import CONFIG

def main():
    """
    Main function to orchestrate the dataset generation process.

    This function performs the following steps:
    1. Loads input data from the specified input folder.
    2. Sets up the necessary language models.
    3. Generates a dataset based on the input texts and models.
    4. Saves the raw generated dataset to a JSONL file.
    5. Validates the generated examples.
    6. Saves the validated dataset to a separate JSONL file.
    """
    print("Loading input data...")
    input_texts = load_input_data(CONFIG['input_folder'])
    if not input_texts:
        print("No valid input files found. Please check your input folder.")
        return

    print("Setting up models...")
    models = setup_models()

    print(f"Generating {CONFIG['num_examples']} examples...")
    dataset = generate_dataset(input_texts, models)

    print(f"Saving raw dataset to {CONFIG['output_file']}...")
    save_to_jsonl(dataset, CONFIG['output_file'])

    print("Validating generated examples...")
    validated_dataset = validate_dataset(dataset, models["sentence_model"])

    print(f"Saving validated dataset to {CONFIG['validated_output_file']}...")
    save_to_jsonl(validated_dataset, CONFIG['validated_output_file'])

    print(f"Raw dataset with {len(dataset)} examples saved to '{CONFIG['output_file']}'")
    print(f"Validated dataset with {len(validated_dataset)} examples saved to '{CONFIG['validated_output_file']}'")

if __name__ == "__main__":
    main()
