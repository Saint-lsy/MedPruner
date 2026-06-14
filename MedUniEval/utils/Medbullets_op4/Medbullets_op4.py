import torch
import os
import json
import gc

from PIL import Image
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt
class Medbullets_op4(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "tuenguyen/Medical-Eval-MedBullets_op4"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

    def load_data(self):
        dataset_path = self.dataset_path
        datasets = load_dataset(self.dataset_path)["train"]

        for idx,sample in tqdm(enumerate(datasets)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        choices = sample["options"]
        choices = [f"{key}.{value}" for key,value in choices.items()]
        
        question = sample["question"]
        answer = sample["answer_idx"]
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)

        messages = {"prompt":prompt}
        sample["messages"] = messages
        sample["answer"] = answer
        sample["choices"] = choices
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        metrics = {}
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            response = extract(response,"answer")
            choices = sample["choices"]
            answer = sample["answer"]
            
            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if correct:
                right += 1
            total += 1
        
        metrics["total metrics"] = {"total":total,"right":right,"acc":right/total}
        
        return metrics,out_samples
