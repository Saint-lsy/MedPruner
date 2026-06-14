import os
import json
from collections import defaultdict
from tqdm import tqdm
import re
from ..base_dataset import BaseDataset
from ..utils import save_json,judge_multi_choice

class Medmnist(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = './medminist_test.json'
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx"))
        self.num_chunks = int(os.environ.get("num_chunks"))

    def load_data(self):

        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Error loading dataset from {self.dataset_path}: {e}")
            return []

        for idx, sample in tqdm(enumerate(dataset), desc=f"Loading data for chunk {self.chunk_idx}"):
            if idx % self.num_chunks == self.chunk_idx:
                prompt_text = sample["conversations"][0]["value"]
                choices_raw = re.findall(r'\n([A-Z]\..*?)(?=\n[A-Z]\.|\Z)', prompt_text, re.DOTALL)
                
 
                choices_clean = [choice.split('.', 1)[1].strip() for choice in choices_raw]
                sample['choices'] = choices_clean 
                processed_sample = self.construct_messages(sample)
                self.samples.append(processed_sample)
                
        print(f"Chunk {self.chunk_idx}/{self.num_chunks} loaded {len(self.samples)} samples.")
        return self.samples

    def construct_messages(self, sample):
   
        prompt = sample["conversations"][0]["value"]+"\nAnswer with the option's letter from the given choices directly."
        prompt=prompt.replace('<image>\n','')
        prompt=prompt.replace('<image>','')
        image_path = sample["images"][0]
        
        messages = {"prompt": prompt, "image": image_path}
        
        sample["messages"] = messages
        if "images" in sample:
            del sample["images"]
        
        return sample

    def cal_metrics(self, out_samples):
        total_count = 0
        total_correct = 0
        split_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        for sample in tqdm(out_samples, desc="Calculating metrics"):
            response = sample.get("response", "")
            answer = sample.get("answer", "")
            split_type = sample.get("split", "unknown")
            choices = sample["choices"]
            is_correct = judge_multi_choice(choices, answer, response)
            
            total_count += 1
            split_stats[split_type]["total"] += 1
            
            if is_correct:
                total_correct += 1
                split_stats[split_type]["correct"] += 1
            
            sample["correct"] = is_correct

        total_acc = total_correct / total_count if total_count > 0 else 0
        metrics = {
            "total_metrics": {
                "total": total_count,
                "correct": total_correct,
                "accuracy": round(total_acc * 100, 2)
            },
            "split_metrics": {}
        }

        for split_name, stats in split_stats.items():
            split_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            metrics["split_metrics"][split_name] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(split_acc * 100, 2)
            }
            
        return metrics, out_samples
                