import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice,judger
from ..base_dataset import BaseDataset
from ..question_formats import get_multiple_choice_prompt

class SuperGPQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "m-a-p/SuperGPQA"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    

    
    def load_data(self):
        dataset_path = self.dataset_path
        dataset = load_dataset(dataset_path)["train"]

        # ['index', 'Figure_path', 'Caption', 'Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D', 'Answer', 'split']
        for idx,sample in tqdm(enumerate(dataset)):
            if sample["discipline"] != "Medicine":
                continue
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        question = sample["question"]
        choices = sample["options"]
        answer = sample["answer_letter"]
        choices = [f"{chr(65+i)}.{choices[i]}" for i in range(len(choices))]

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)

        messages = {"prompt":prompt}
        sample["messages"] = messages
        sample["choices"] = choices
        sample["answer"] = answer
        del sample["answer_letter"]
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
        total_field_dict = defaultdict(int)
        right_field_dict = defaultdict(int)
        total_difficulty_dict = defaultdict(int)
        right_difficulty_dict = defaultdict(int)
        for i,sample in enumerate(out_samples):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]
            field = sample["field"]
            difficulty = sample["difficulty"]
            response = extract(response,"answer")
            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct
            if correct:
                right_field_dict[field] += 1
                right_difficulty_dict[difficulty] += 1
                right += 1
            total_difficulty_dict[difficulty] += 1
            total_field_dict[field] += 1
            total += 1
        field_metrics = {}
        difficulty_metrics = {}
        for key,value in total_field_dict.items():
            right_cnt = right_field_dict[key]
            difficulty_metrics[key] = {"total":value,"right":right_cnt,"acc":right_cnt/value}
        
        for key,value in total_difficulty_dict.items():
            right_cnt = right_difficulty_dict[key]
            field_metrics[key] = {"total":value,"right":right_cnt,"acc":right_cnt/value}

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total},"field":field_metrics,"difficulty":difficulty_metrics}
        return metrics,out_samples



                