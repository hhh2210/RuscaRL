import os
import json
import datasets
import pandas as pd
import random
from typing import List, Dict, Any

def load_gpqa_csv(file_path: str) -> List[Dict[str, Any]]:
    """Load GPQA Diamond CSV file"""
    df = pd.read_csv(file_path)
    data = []
    
    for _, row in df.iterrows():
        # Load only necessary fields: question and options
        question = row['Question']
        correct_answer = row['Correct Answer']
        incorrect_1 = row['Incorrect Answer 1']
        incorrect_2 = row['Incorrect Answer 2']
        incorrect_3 = row['Incorrect Answer 3']
        
        # Skip any rows with missing key fields
        if pd.isna(question) or pd.isna(correct_answer) or pd.isna(incorrect_1) or pd.isna(incorrect_2) or pd.isna(incorrect_3):
            continue
        
        data.append({
            'question': str(question),
            'correct_answer': str(correct_answer),
            'incorrect_answers': [str(incorrect_1), str(incorrect_2), str(incorrect_3)]
        })
    
    return data



def shuffle_options(correct_answer: str, incorrect_answers: List[str], target_position: str) -> Dict[str, str]:
    """
    Rearrange options so that the correct answer appears at the specified position
    target_position: 'A', 'B', 'C', or 'D'
    """
    all_answers = [correct_answer] + incorrect_answers
    random.shuffle(all_answers)
    
    # Ensure correct answer is at target position
    options = {'A': '', 'B': '', 'C': '', 'D': ''}
    positions = ['A', 'B', 'C', 'D']
    
    # Place correct answer at target position
    options[target_position] = correct_answer
    
    # Randomly assign other answers to remaining positions
    remaining_positions = [pos for pos in positions if pos != target_position]
    random.shuffle(remaining_positions)
    
    for i, answer in enumerate(incorrect_answers):
        options[remaining_positions[i]] = answer
    
    return options, target_position

def make_map_fn(split: str):
    """Construct data mapping function"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Construct multiple choice format
        GPQA_TEMPLATE = "{Question}\n\n\nA) {A}\nB) {B}\nC) {C}\n\nD) {D}"
        
        # Get options and correct answer position
        options, correct_position = example['options'], example['correct_position']
        
        question_text = GPQA_TEMPLATE.format(
            Question=example['question'],
            A=options['A'],
            B=options['B'], 
            C=options['C'],
            D=options['D']
        )
        
        # Add instruction
        instruction_following = "Let's think step by step and output the final answer as a single option letter (A, B, C, or D) within \\boxed{}."
        prompt_content = question_text + "\n\n" + instruction_following
        
        prompt = [{"role": "user", "content": prompt_content}]
        
        # Construct rubrics using dict format tags
        rubrics = [
            {
                "criterion": f"The answer is {correct_position}",
                "points": 10.0,
                "tags": {
                    "function": "math_verify",
                    "parameters": {
                        "answer": correct_position
                    },
                    "verifier": "rule"
                }
            }
        ]
        
        # Construct reward_model field
        reward_model = {
            "style": "rubric",
            "rubrics": rubrics,
            "ground_truth": correct_position
        }
        
        # Construct data format required by verl
        data = {
            "prompt": prompt,
            "data_source": "gpqa_diamond", 
            "ability": "science",
            "reward_model": reward_model,
            "extra_info": {
                "prompt": prompt,
                "reward_model": reward_model
            }
        }
        return data
    
    return process_fn

def create_four_versions(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create four versions for each question, with correct answer at A, B, C, D positions respectively
    Maintain original order, with 4 versions of the same question arranged in A, B, C, D order
    """
    expanded_data = []
    
    for item in data_list:
        # Create 4 versions for each question in A, B, C, D order
        for target_pos in ['A', 'B', 'C', 'D']:
            # Create one version for each position
            options, correct_position = shuffle_options(
                item['correct_answer'], 
                item['incorrect_answers'], 
                target_pos
            )
            
            new_item = item.copy()
            new_item['options'] = options
            new_item['correct_position'] = correct_position
            expanded_data.append(new_item)
    
    return expanded_data

def process_dataset(data_list: List[Dict[str, Any]], split: str) -> datasets.Dataset:
    """Process dataset, maintaining original order"""
    # First create four versions, maintaining order
    expanded_data = create_four_versions(data_list)
    
    dataset = datasets.Dataset.from_list(expanded_data)
    processed_dataset = dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names  # Remove all original columns
    )
    
    # Don't shuffle, maintain original order
    return processed_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='raw_data/GPQA-D/gpqa_diamond.csv')
    parser.add_argument('--output_dir', default='data/gpqa_diamond')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(42)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_gpqa_csv(args.input_file)
    print(f"Loaded {len(data)} samples")
    
    # Process test set (create four versions)
    test_dataset = process_dataset(data, 'test')
    
    # Save as parquet format
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use pyarrow to save, avoiding huggingface metadata issues
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Convert to pandas DataFrame then to pyarrow table, without huggingface metadata
    df = test_dataset.to_pandas()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, os.path.join(args.output_dir, 'test.parquet'))
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Original data: {len(data)} questions")
    print(f"Expanded test set size: {len(test_dataset)} (4 versions per question)")
    
    # Verify correct answer distribution
    correct_positions = [example['reward_model']['ground_truth'] for example in test_dataset]
    position_counts = {pos: correct_positions.count(pos) for pos in ['A', 'B', 'C', 'D']}
    print(f"Correct answer position distribution: {position_counts}")
    
    # Print data sample example
    print("\nTest set sample example:")
    print(json.dumps(test_dataset[0], indent=2, ensure_ascii=False))
    
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main()