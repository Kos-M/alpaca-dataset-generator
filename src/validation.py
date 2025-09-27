from typing import List, Dict, Any
from tqdm import tqdm
from utils import is_valid_output

def validate_dataset(dataset: List[Dict[str, Any]], sentence_model) -> List[Dict[str, Any]]:
    """
    Validates a generated dataset of instruction-following examples.

    This function iterates through each example in the provided dataset and applies
    a series of validation checks using the `is_valid_output` utility function.
    Only examples that pass all validation criteria are included in the returned
    validated dataset.

    Args:
        dataset (List[Dict[str, Any]]): The raw dataset containing generated examples.
                                       Each example is a dictionary with keys like
                                       "instruction_type", "output", and "input".
        sentence_model: The sentence embedding model used for similarity-based validation.

    Returns:
        List[Dict[str, Any]]: A new list containing only the examples that passed validation.
    """
    validated_dataset = []
    print(f"Starting validation of {len(dataset)} examples")
    
    with tqdm(total=len(dataset), desc="Validating examples", unit="example") as pbar:
        for i, example in enumerate(dataset):
            if is_valid_output(example["instruction_type"], example["output"], example["input"], sentence_model):
                validated_dataset.append(example)
            else:
                print(f"Example {i} failed validation")
            pbar.update(1)
    
    print(f"Validation complete. {len(validated_dataset)} out of {len(dataset)} examples passed validation")
    return validated_dataset
