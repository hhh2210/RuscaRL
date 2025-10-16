import os
import json
import datasets
import pandas as pd
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def make_map_fn(split: str):
    """Construct data mapping function"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract problem
        problem = example['problem']
        answer = example['answer']
        
        # Construct prompt format
        instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
        prompt_content = problem + " " + instruction_following
        prompt = [{"role": "user", "content": prompt_content}]
        
        # Construct rubrics using dict format tags
        rubrics = [
            {
                "criterion": f"The answer is {answer}",
                "points": 10.0,
                "tags": {
                    "function": "math_verify",
                    "parameters": {
                        "answer": answer
                    },
                    "verifier": "rule"
                }
            }
        ]
        
        # Construct reward_model field
        reward_model = {
            "style": "rubric",
            "rubrics": rubrics,
            "ground_truth": answer
        }
        
        # Construct data format required by verl
        data = {
            "prompt": prompt,
            "data_source": "math500",
            "ability": "math",
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
    return processed_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='raw_data/Math500/math500.jsonl')
    parser.add_argument('--output_file', default='data/Math500/math500-rubrics.parquet')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} samples")
    
    # Process dataset
    processed_dataset = process_dataset(data, 'train')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Use pyarrow to save, avoiding huggingface metadata issues
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    # Convert to pandas DataFrame then to pyarrow table, without huggingface metadata
    df = processed_dataset.to_pandas()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, args.output_file)
    
    print(f"\n✅ Data processing completed!")
    print(f"Output file: {args.output_file}")
    print(f"Processed data count: {len(processed_dataset)}")
    
    # Print data sample example
    print("\n📋 Data sample example:")
    sample = processed_dataset[0]
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    
    print(f"\n📊 Data statistics:")
    print(f"   Total samples: {len(processed_dataset)}")
    print(f"   Data source: math500")
    print(f"   Format: rubrics")

if __name__ == '__main__':
    main()