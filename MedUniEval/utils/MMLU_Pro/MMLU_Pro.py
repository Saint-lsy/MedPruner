
import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from ..base_dataset import BaseDataset
from ..utils import save_json,extract,judge_multi_choice

medical_subject = ["anatomy","clinical_knowledge","college_biology","college_medicine","medical_genetics","professional_medicine","nutrition","virology","high_school_biology"]

class MMLU_Pro(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        super().__init__() 
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path 
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))
    
    def load_data(self):
        datas = load_dataset('json', data_files="./mmlu_pro_test.json")["train"]

        dataset = [data for data in datas]

        for idx, sample in enumerate(tqdm(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                processed_sample = self.construct_messages(sample)
                self.samples.append(processed_sample)
        return self.samples

    def construct_messages(self, sample):

        choices_dict = sample["choices"]
        choices_str = "\n".join([f"{key}. {value}" for key, value in choices_dict.items()])

        question = sample["problem"]
 
        answer_letter = sample["answer"]

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        if is_reasoning:
            prompt = f"""{question}\n{choices_str}\nPlease reason step by step, and put your final answer within \boxed{{}}."""
        else:
            prompt = f"{question}\n{choices_str}\nAnswer with the option's letter from the given choices directly."      

        sample["prompt"] = prompt
        sample["messages"] = {"prompt": prompt} 
        sample["answer_letter"] = answer_letter
        
        return sample


    def cal_metrics(self, out_samples):
 
        total = 0
        right = 0
        for i, sample in enumerate(out_samples):
            response = sample["response"] 
            response_letter = extract(response, "answer")

            correct_letter = sample["answer_letter"] 
            
            is_correct = judge_multi_choice(sample["choices"], correct_letter, response_letter)
            
            out_samples[i]["correct"] = is_correct
            out_samples[i]["response_letter"] = response_letter

            if is_correct:
                right += 1
            total += 1
        
        accuracy = right / total if total > 0 else 0
        metrics = {"total metrics": {"total": total, "right": right, "acc": accuracy}}
        return metrics, out_samples
                

