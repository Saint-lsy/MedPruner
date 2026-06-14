import torch
import os
import json
import gc
import csv

from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm

from ..utils import save_json,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import *

class MedFrameQA(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "SuhaoYu1020/MedFrameQA"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))


    def run(self,samples,model,batch_size = 500):
      out_samples = []
      with torch.no_grad():
          messages_list = []
          current_messages = []
          current_samples = []
          for sample in tqdm(samples, desc="Batching Samples"):
              messages = sample["messages"]
              current_messages.append(messages)
              current_samples.append(sample)
              if len(current_messages) >= batch_size:
                  messages_list.append([current_messages,current_samples])
                  current_messages = []
                  current_samples = []
          if current_messages:
              messages_list.append([current_messages,current_samples])
          
          for current_messages,current_samples in tqdm(messages_list, desc="Running Inference"):
              outputs = model.generate_outputs(current_messages)
              try:
                  for sample,response in zip(current_samples,outputs):
                      del sample["messages"]
                      sample["response"] = response
                      out_samples.append(sample)   
              except Exception as e:
                  from pdb import set_trace;set_trace()
                  print(e)
              gc.collect()
      return out_samples
    
    def load_data(self):
        dataset = load_dataset(self.dataset_path)["test"]
        for idx,sample in tqdm(enumerate(dataset), total=len(dataset), desc="Loading Data"):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples


    def construct_messages(self,sample):
  
        question = sample["question"]
        choices = sample["options"]
        answer = sample["correct_answer"]
        image_1 = sample["image_1"]
        image_2 = sample["image_2"]
        image_3 = sample["image_3"]
        image_4 = sample["image_4"]
        image_5 = sample["image_5"]

        choices_formatted = [f"{chr(65+i)}.{choices[i]}" for i in range(len(choices))]

        images = [image_1,image_2,image_3,image_4,image_5]
        images = [image for image in images if image is not None]

        sample['frame_count'] = len(images)

        is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
        prompt = get_multiple_choice_prompt(question,choices_formatted,is_reasoning)

        messages = {"prompt":prompt,"images":images}
        sample["messages"] = messages
        sample["choices"] = choices_formatted
        sample["answer"] = answer
        
        del sample["options"]
        del sample["correct_answer"]
        del sample["image_1"]
        del sample["image_2"]
        del sample["image_3"]
        del sample["image_4"]
        del sample["image_5"]
        
        return sample


    def cal_metrics(self,out_samples):

        total = 0
        right = 0
        
        total_by_frame_count = defaultdict(int)
        right_by_frame_count = defaultdict(int)
        
        total_by_modality = defaultdict(int)
        right_by_modality = defaultdict(int)

        primary_modalities = ['CT', 'MRI', 'ultrasound', 'X-ray']

        for i,sample in enumerate(tqdm(out_samples, desc="Calculating Metrics")):
            response = sample["response"]
            choices = sample["choices"]
            answer = sample["answer"]

            frame_count = sample['frame_count']
            modality_raw = sample['modality'] 
            modality = modality_raw if modality_raw in primary_modalities else 'Other'

            correct = judge_multi_choice(choices,answer,response)
            out_samples[i]["correct"] = correct

            total += 1
            total_by_frame_count[frame_count] += 1
            total_by_modality[modality] += 1

            if correct:
                right += 1
                right_by_frame_count[frame_count] += 1
                right_by_modality[modality] += 1

        metrics = {
            "total_metrics": {
                "total": total,
                "right": right,
                "accuracy": (right / total) * 100 if total > 0 else 0.0
            }
        }
        
        frame_count_metrics = {}
        for fc in sorted(total_by_frame_count.keys()):
            total_fc = total_by_frame_count[fc]
            right_fc = right_by_frame_count[fc]
            frame_count_metrics[fc] = {
                "total": total_fc,
                "right": right_fc,
                "accuracy": (right_fc / total_fc) * 100 if total_fc > 0 else 0.0
            }
        metrics["frame_count_metrics"] = frame_count_metrics
        
        modality_metrics = {}
        for mod_key in primary_modalities + ['Other']:
            if mod_key in total_by_modality:
                total_mod = total_by_modality[mod_key]
                right_mod = right_by_modality[mod_key]
                modality_metrics[mod_key] = {
                    "total": total_mod,
                    "right": right_mod,
                    "accuracy": (right_mod / total_mod) * 100 if total_mod > 0 else 0.0
                }
        metrics["modality_metrics"] = modality_metrics

        return metrics, out_samples
