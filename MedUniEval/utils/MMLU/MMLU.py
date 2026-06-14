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
medical_subject = ["anatomy","clinical_knowledge","college_biology","college_medicine","medical_genetics","professional_medicine","nutrition","virology","high_school_biology"]

class MMLU(BaseDataset):
    def __init__(self,model,dataset_path,output_path,subject = "all"):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "cais/mmlu"
        self.samples = []
        self.subject = subject
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    
    def load_data(self):
        dataset = []
        if self.subject == "all":
            datas = load_dataset(self.dataset_path,"all")["test"]
            for data in datas:
                dataset.append(data)
        elif self.subject == "medical":
            for split in medical_subject:
                datas = load_dataset(self.dataset_path,split)["test"]
                for data in datas:
                    dataset.append(data)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples

    def construct_messages(self,sample):
        choices = sample["choices"]
        answer = sample["answer"]
        question = sample["question"]
        alphas = ["A","B","C","D"]

        choices = "\n".join([f"{alpha}.{choice}" for alpha,choice in zip(alphas,choices)])
        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = f"Context: {context}" + "\n" + get_multiple_choice_prompt(question,choices,is_reasoning)

        messages = {"prompt":prompt}
        sample["prompt"] = prompt
        sample["messages"] = messages
        sample["answer"] = alphas[answer]
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



                