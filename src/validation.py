from typing import List, Dict, Any
from tqdm import tqdm
from utils import is_valid_output

def validate_dataset(dataset: List[Dict[str, Any]], sentence_model) -> List[Dict[str, Any]]:
    """
    Validates a list of generated dataset examples based on predefined criteria.

    This function iterates through each example in the provided `dataset`.
    For each example, it calls `is_valid_output` (from `utils.py`) to check if the
    generated output meets the quality and relevance standards. Examples that pass
    validation are included in the returned list.

    Args:
        dataset (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                       represents a generated example with at least
                                       'instruction_type', 'output', and 'input' keys.
        sentence_model: The sentence embedding model used for similarity checks within validation.

    Returns:
        List[Dict[str, Any]]: A new list containing only the examples that passed all validation checks.
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
