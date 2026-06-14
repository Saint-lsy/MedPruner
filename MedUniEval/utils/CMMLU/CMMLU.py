import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt
medical_subject = ["anatomy","clinical_knowledge","college_medicine","genetics","traditional_chinese_medicine","nutrition","virology"]

class CMMLU(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "haonan-li/cmmlu"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        dataset = load_dataset(self.dataset_path)["train"]

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                if sample["task"] in medical_subject:
                    sample = self.construct_messages(sample)
                    self.samples.append(sample)
        print(len(self.samples))
        return self.samples

    def construct_messages(self,sample):
        answer = sample["answer"]
        

        OptionA = sample["A"]
        OptionB = sample["B"]
        OptionC = sample["C"]
        OptionD = sample["D"]
        question = sample["question"]
        choices = [OptionA,OptionB,OptionC,OptionD]
        choices = [f"{chr(65+i)}.{choices[i]}" for i in range(len(choices))]
        
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning,lang = "zh")

        messages = {"prompt":prompt}
        sample["prompt"] = prompt
        sample["messages"] = messages
        sample["answer"] = answer
        sample["choices"] = choices
        # import pdb;pdb.set_trace()
        return sample


    def cal_metrics(self,out_samples):
        total = 0
        right = 0
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

        metrics = {"total metrics":{"total":total,"right":right,"acc":right/total}}
        return metrics,out_samples

