import os
import json
import datasets
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: Dict[str, Any]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d.get("tags", {})
        )

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags
        }

def load_parquet(file_path: str) -> List[Dict[str, Any]]:
    """Load Parquet file"""
    df = pd.read_parquet(file_path)
    # Convert DataFrame to list of dictionaries
    data = df.to_dict('records')
    return data

def convert_rar_to_message_format(question_text: str) -> List[Dict[str, str]]:
    """Convert RaR dataset question text to message format"""
    # RaR dataset question is plain text, needs to be converted to message list format
    return [
        {
            "content": question_text,
            "role": "user"
        }
    ]

def convert_rubric_to_rubric_items(rubric_list: List[Dict[str, Any]]) -> List[RubricItem]:
    """Convert RaR dataset rubric to RubricItem format"""
    rubric_items = []
    
    for rubric in rubric_list:
        # Map description to criterion, weight to points
        criterion = rubric.get('description', '')
        points = rubric.get('weight', 0.0)
        
        # RaR dataset uses LLM verification, set verifier field and source field
        tags = {"verifier": "llm"}
        
        rubric_item = RubricItem(
            criterion=criterion,
            points=points,
            tags=tags
        )
        rubric_items.append(rubric_item)
    
    return rubric_items

def make_map_fn(data_source: str):
    """Construct data mapping function"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract question and convert to message format
        question_text = example['question']
        prompt = convert_rar_to_message_format(question_text)
        
        # Extract rubrics and convert to RubricItem format
        rubric_list = example['rubric']
        rubric_items = convert_rubric_to_rubric_items(rubric_list)
        
        # Construct reward_model field
        reward_model = {
            "style": "rubric",
            "rubrics": [r.to_dict() for r in rubric_items],
            "ground_truth": ""  # Use empty string
        }
        
        # Construct data format required by verl
        data = {
            "data_source": data_source,
            "prompt": prompt,  # Keep outer prompt
            "ability": "reasoning_chat",  # RaR dataset is mainly reasoning tasks
            "reward_model": reward_model,  # Keep outer reward_model
            "extra_info": {
                "prompt": prompt,  # Also put prompt in extra_info
                "reward_model": reward_model  # Also put reward_model in extra_info
            }
        }
        return data
    
    return process_fn

def process_dataset(data_list: List[Dict[str, Any]], data_source: str) -> datasets.Dataset:
    """Process dataset"""
    dataset = datasets.Dataset.from_list(data_list)
    processed_dataset = dataset.map(
        function=make_map_fn(data_source),
        with_indices=True,
        remove_columns=dataset.column_names  # Remove all original columns
    )
    # Shuffle the data
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    return shuffled_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/RaR')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    # Load RaR-Medicine data
    medicine_file = 'raw_data/RaR/RaR-Medicine-20k-o3-mini.parquet'
    medicine_data = load_parquet(medicine_file)
    
    # Load RaR-Science data
    science_file = 'raw_data/RaR/RaR-Science-20k-o3-mini.parquet'
    science_data = load_parquet(science_file)
    
    # Process datasets
    medicine_dataset = process_dataset(medicine_data, 'RaR-Medicine')
    science_dataset = process_dataset(science_data, 'RaR-Science')
    
    # Save as parquet format
    os.makedirs(args.output_dir, exist_ok=True)
    medicine_dataset.to_parquet(os.path.join(args.output_dir, 'rar_medicine.parquet'))
    science_dataset.to_parquet(os.path.join(args.output_dir, 'rar_science.parquet'))
    
    # Print dataset information
    print("\nDataset information:")
    print(f"RaR-Medicine dataset size: {len(medicine_dataset)}")
    print(f"RaR-Science dataset size: {len(science_dataset)}")
    
    # Print sample examples
    print("\nRaR-Medicine sample example:")
    print(json.dumps(medicine_dataset[0], indent=2, ensure_ascii=False))
    print("\nRaR-Science sample example:")
    print(json.dumps(science_dataset[0], indent=2, ensure_ascii=False))
    
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main()