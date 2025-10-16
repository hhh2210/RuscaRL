import os
import json
import datasets
import random
import re
from typing import List, Dict, Any, Set

def load_researchqa_json(file_path: str) -> List[Dict[str, Any]]:
    """Load ResearchQA JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_train_data(full_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]], valid_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get train data by removing test and valid data from full data"""
    # Create sets of IDs from test and valid data for efficient lookup
    test_ids: Set[str] = {item['id'] for item in test_data}
    valid_ids: Set[str] = {item['id'] for item in valid_data}
    
    # Filter out test and valid data from full data
    train_data = []
    for item in full_data:
        if item['id'] not in test_ids and item['id'] not in valid_ids:
            train_data.append(item)
    
    return train_data


def get_eval_data(test_data: List[Dict[str, Any]], num_samples: int = 500) -> List[Dict[str, Any]]:
    """Randomly select samples from test data for evaluation"""
    if len(test_data) <= num_samples:
        return test_data.copy()
    
    # Randomly sample without replacement
    return random.sample(test_data, num_samples)

def convert_question_to_statement(question: str) -> str:
    """Convert question format to statement format for rubric items"""
    question = question.strip()
    
    # Pattern 1: "Does the response..." -> "The response..."
    if question.startswith("Does the response"):
        # Remove "Does " and convert to statement
        statement = question[5:]  # Remove "Does "
        # Remove question mark if present
        if statement.endswith("?"):
            statement = statement[:-1]
        # Ensure proper capitalization and punctuation
        statement = statement[0].upper() + statement[1:] if statement else ""
        if statement and not statement.endswith('.'):
            statement += '.'
        return statement
    
    # Pattern 2: "Does it..." -> "It..."
    elif question.startswith("Does it"):
        # Remove "Does " and convert to statement
        statement = question[5:]  # Remove "Does "
        # Remove question mark if present
        if statement.endswith("?"):
            statement = statement[:-1]
        # Ensure proper capitalization and punctuation
        statement = statement[0].upper() + statement[1:] if statement else ""
        if statement and not statement.endswith('.'):
            statement += '.'
        return statement
    
    # Pattern 3: "Does the explanation..." -> "The explanation..."
    elif question.startswith("Does the explanation"):
        # Remove "Does " and convert to statement
        statement = question[5:]  # Remove "Does "
        # Remove question mark if present
        if statement.endswith("?"):
            statement = statement[:-1]
        # Ensure proper capitalization and punctuation
        statement = statement[0].upper() + statement[1:] if statement else ""
        if statement and not statement.endswith('.'):
            statement += '.'
        return statement
    
    # Pattern 4: "Is there..." -> "There is..."
    elif question.startswith("Is there"):
        # Convert "Is there" to "There is"
        statement = "There is" + question[8:]  # Replace "Is there" with "There is"
        # Remove question mark if present
        if statement.endswith("?"):
            statement = statement[:-1]
        # Ensure proper punctuation (capitalization already correct)
        if statement and not statement.endswith('.'):
            statement += '.'
        return statement
    
    # For any other patterns, try a general approach
    else:
        # Remove question mark if present
        statement = question
        if statement.endswith("?"):
            statement = statement[:-1]
        # Ensure proper capitalization and punctuation
        if statement:
            statement = statement[0].upper() + statement[1:] if len(statement) > 1 else statement.upper()
            if not statement.endswith('.'):
                statement += '.'
        return statement

def make_map_fn(split: str, use_statements: bool = False):
    """Construct data mapping function
    
    Args:
        split: Dataset split name
        use_statements: If True, convert questions to statements; if False, keep original questions
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract query and convert to message format
        query = example['query']
        prompt = [{"role": "user", "content": query}]
        
        # Convert each rubric_item to a criterion with proper format
        rubrics = []
        for rubric in example.get('rubric', []):
            rubric_item = rubric.get('rubric_item', '')
            if rubric_item.strip():
                # Convert to statement if requested
                if use_statements:
                    criterion = convert_question_to_statement(rubric_item)
                else:
                    criterion = rubric_item
                
                rubrics.append({
                    "criterion": criterion,
                    "points": 10.0,
                    "tags": {
                        "function": "llm",
                        "parameters": {
                            "answer": ""
                        },
                        "verifier": "llm"
                    }
                })
        
        # If no rubrics found, raise an error
        if not rubrics:
            raise ValueError(f"No rubrics found for item {example.get('id', 'unknown')}")
        
        # Construct reward_model field
        reward_model = {
            "style": "rubric",
            "rubrics": rubrics,
            "ground_truth": ""
        }
        
        # Construct data format required by verl
        data = {
            "prompt": prompt,
            "data_source": "ResearchQA",
            "ability": "science",
            "reward_model": reward_model,
            "extra_info": {
                "prompt": prompt,
                "reward_model": reward_model
            }
        }
        return data
    
    return process_fn

def process_dataset(data_list: List[Dict[str, Any]], split: str) -> datasets.Dataset:
    """Process dataset"""
    dataset = datasets.Dataset.from_list(data_list)
    processed_dataset = dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names  # Remove all original columns
    )
    # Shuffle the data
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    return shuffled_dataset

def process_dataset_with_statements(data_list: List[Dict[str, Any]], split: str, use_statements: bool = False) -> datasets.Dataset:
    """Process dataset with option to convert questions to statements"""
    dataset = datasets.Dataset.from_list(data_list)
    processed_dataset = dataset.map(
        function=make_map_fn(split, use_statements),
        with_indices=True,
        remove_columns=dataset.column_names  # Remove all original columns
    )
    # Shuffle the data
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    return shuffled_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='raw_data/ResearchQA')
    parser.add_argument('--output_dir', default='data/ResearchQA')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Loading ResearchQA data...")
    full_data = load_researchqa_json(os.path.join(args.local_dir, "full.json"))
    test_data = load_researchqa_json(os.path.join(args.local_dir, "test.json"))
    valid_data = load_researchqa_json(os.path.join(args.local_dir, "valid.json"))
    
    print(f"Loaded {len(full_data)} examples from full.json")
    print(f"Loaded {len(test_data)} examples from test.json")
    print(f"Loaded {len(valid_data)} examples from valid.json")
    
    # Get train data by removing test and valid from full
    train_data = get_train_data(full_data, test_data, valid_data)
    print(f"Train set: {len(train_data)} examples (full - test - valid)")
    
    # Get eval data by randomly sampling from test
    eval_data = get_eval_data(test_data, num_samples=500)
    print(f"Eval set: {len(eval_data)} examples (randomly sampled from test)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process both versions: original and statements
    for version_name, use_statements in [("original", False), ("statements", True)]:
        print(f"\nProcessing {version_name} version...")
        
        # Process training and evaluation sets
        train_dataset = process_dataset_with_statements(train_data, 'train', use_statements)
        eval_dataset = process_dataset_with_statements(eval_data, 'eval', use_statements)
        
        # Determine file suffix
        suffix = "_original" if version_name == "original" else ""
        
        # Save as parquet format
        train_dataset.to_parquet(os.path.join(args.output_dir, f'train{suffix}.parquet'))
        eval_dataset.to_parquet(os.path.join(args.output_dir, f'eval{suffix}.parquet'))
        
        print(f"  {version_name.capitalize()} version saved:")
        print(f"    Training set size: {len(train_dataset)}")
        print(f"    Evaluation set size: {len(eval_dataset)}")
    
    # Print data sample examples (statements version)
    print("\nSample statements version data:")
    sample_dataset = process_dataset_with_statements(train_data[:1], 'train', True)
    print(json.dumps(sample_dataset[0], indent=2, ensure_ascii=False))
    
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main()