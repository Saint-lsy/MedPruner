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

from ..question_formats import get_multiple_choice_prompt

class MedQA_USMLE(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "GBaker/MedQA-USMLE-4-options"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        dataset = load_dataset(self.dataset_path)["test"]
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        question = sample["question"]
        answer = sample["answer_idx"]
        options = sample["options"]
        choiceA = options["A"]
        choiceB = options["B"]
        choiceC = options["C"]
        choiceD = options["D"]
        choiceA = f"A. {choiceA}"
        choiceB = f"B. {choiceB}"
        choiceC = f"C. {choiceC}"
        choiceD = f"D. {choiceD}"
        choices = [choiceA,choiceB,choiceC,choiceD]
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices,is_reasoning)

        sample = {"prompt":prompt,"answer":answer,"choices":choices,"messages":{"prompt":prompt}}
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

