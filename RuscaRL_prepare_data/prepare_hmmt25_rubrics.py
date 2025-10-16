import os
import json
import datasets
import pandas as pd
from typing import List, Dict, Any

def load_parquet(file_path: str) -> List[Dict[str, Any]]:
    """Load Parquet file"""
    df = pd.read_parquet(file_path)
    # Convert DataFrame to list of dictionaries
    data = df.to_dict('records')
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
            "data_source": "hmmt25",
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
    """Process dataset with 8x data augmentation"""
    # Replicate original data 8 times
    expanded_data_list = data_list * 8
    
    dataset = datasets.Dataset.from_list(expanded_data_list)
    processed_dataset = dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names  # Remove all original columns
    )
    return processed_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='raw_data/HMMT/hmmt25.parquet')
    parser.add_argument('--output_file', default='data/HMMT/hmmt25-rubrics.parquet')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_parquet(args.input_file)
    print(f"Loaded {len(data)} original samples")
    print(f"Will expand to {len(data) * 8} samples (8x augmentation)")
    
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
    print(f"   Original samples: {len(data)}")
    print(f"   Augmented samples: {len(processed_dataset)} (8x augmentation)")
    print(f"   Data source: hmmt25")
    print(f"   Format: rubrics")

if __name__ == '__main__':
    main()