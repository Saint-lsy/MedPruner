import torch
import os
import json
import gc
import csv
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from PIL import Image
from datasets import load_dataset,load_from_disk
from collections import defaultdict
from tqdm import tqdm
from .utils import RubricItem,GRADER_TEMPLATE,calculate_score,_aggregate_get_clipped_mean,parse_json_to_dict
from ..utils import save_json,extract,judger,deal_tasks
from ..base_dataset import BaseDataset
import concurrent.futures

def read_jsonl(jsonl_path):
    new_datas = []
    with open(jsonl_path,"r") as f:
            datas = f.readlines()
    for line in datas:
        data = json.loads(line)
        new_datas.append(data)
    return new_datas
class HealthBench(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))

    def load_data(self):
        dataset_path = "./HealthBench"
        dataset = []
        consensus_path = os.path.join(dataset_path,"consensus_2025-05-09-20-00-46.jsonl")
        hard_path = os.path.join(dataset_path,"hard_2025-05-08-21-00-10.jsonl")
        eval_path = os.path.join(dataset_path,"2025-05-07-06-14-12_oss_eval.jsonl")
        consensus_dataset = read_jsonl(consensus_path)
        hard_dataset = read_jsonl(hard_path)
        eval_dataset = read_jsonl(eval_path)
        for data in eval_dataset:
            data["dataset_type"] = "normal"
            dataset.append(data)
        for data in consensus_dataset:
            data["dataset_type"] = "consensus"
            dataset.append(data)
        for data in hard_dataset:
            data["dataset_type"] = "hard"
            dataset.append(data)

        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                sample = self.construct_messages(sample)
                self.samples.append(sample)
        return self.samples


    def construct_messages(self, sample):
        prompt = sample['prompt'][0]['content']
        messages = {"prompt":prompt}
        sample["messages"] = messages
        return sample
        
    def _grade_single_rubric_with_retry(self, task):
        attempt = 0
        while True:
            attempt += 1
            try:
                raw_response = judger.response(task["messages"], temperature=0.0)

                if not raw_response:
                    raise ValueError("API returned an empty response.")
                
                parsed_grade = parse_json_to_dict(raw_response)
                
                if "error" in parsed_grade or "criteria_met" not in parsed_grade or not isinstance(parsed_grade["criteria_met"], bool):
                    error_msg = parsed_grade.get("error", "Invalid JSON format.")
                    raise ValueError(f"Response parsing failed: {error_msg}")
                
                return task, parsed_grade

            except Exception as e:
                print(f"Task for sample {task['sample_index']}, rubric {task['rubric_index']} failed on attempt {attempt}. Error: {e}. Retrying in 1s...")
                time.sleep(1) 


    def cal_metrics(self, out_samples):
        all_grading_tasks = []
        for sample_index, sample in enumerate(out_samples):
            prompt = sample["prompt"]
            response = sample["response"]
            rubrics = sample["rubrics"]
            
            conversations_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in (prompt + [dict(content=response, role="assistant")])]
            )
            
            for rubric_index, rubric_dict in enumerate(rubrics):
                rubric_item = RubricItem.from_dict(rubric_dict)
                grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversations_str).replace("<<rubric_item>>", str(rubric_item))
                
                task = {
                    "sample_index": sample_index,
                    "rubric_index": rubric_index,
                    "messages": [dict(content=grader_prompt, role="user")]
                }
                all_grading_tasks.append(task)
        
        grading_results = {} 
        with ThreadPoolExecutor(max_workers=128) as executor:
            future_to_task_info = {
                executor.submit(self._grade_single_rubric_with_retry, task): task
                for task in all_grading_tasks
            }
            
            for future in tqdm(as_completed(future_to_task_info), total=len(all_grading_tasks), desc="Grading All Rubrics (with retry)"):
                try:
                    completed_task, grade = future.result()
                    sample_idx = completed_task["sample_index"]
                    rubric_idx = completed_task["rubric_index"]
                    grading_results[(sample_idx, rubric_idx)] = grade
                except Exception as e:
                    task_info = future_to_task_info[future]
                    print(f"A task for sample {task_info['sample_index']} failed catastrophically: {e}")
                    grading_results[(task_info['sample_index'], task_info['rubric_index'])] = {
                        "criteria_met": False,
                        "explanation": f"Grading failed catastrophically: {e}"
                    }

        processed_samples = []
        for sample_index, sample in enumerate(out_samples):
            rubrics = sample["rubrics"]
            rubric_items = [RubricItem.from_dict(d) for d in rubrics]
            
            grading_response_list = [grading_results.get((sample_index, i)) for i in range(len(rubrics))]
            
            if any(g is None for g in grading_response_list):
                sample["metrics"] = None
                sample["readable_explanation_str"] = "Could not grade this sample due to some failed rubric gradings."
                sample["rubric_items_with_grades"] = None
                processed_samples.append(sample)
                continue
            
            example_tags = sample["example_tags"]
            overall_score = calculate_score(rubric_items, grading_response_list)
            
            if overall_score is None:
                sample["metrics"] = None
            else:
                metrics = {"overall_score": overall_score}
                metrics.update({tag: overall_score for tag in example_tags})

                rubric_tag_items_grades = defaultdict(list)
                for rubric_item, grading_response in zip(rubric_items, grading_response_list):
                    for tag in rubric_item.tags:
                        rubric_tag_items_grades[tag].append((rubric_item, grading_response))

                rubric_tag_scores = {}
                for tag, items_grades in rubric_tag_items_grades.items():
                    items, grades = zip(*items_grades)
                    score = calculate_score(items, grades)
                    if score is not None:
                        rubric_tag_scores[tag] = score
                metrics.update(rubric_tag_scores)
                sample["metrics"] = metrics

            rubric_items_with_grades = []
            readable_explanation_list = []
            for rubric_item, grading_response in zip(rubric_items, grading_response_list):
                explanation = grading_response.get("explanation", "No explanation provided")
                criteria_met = grading_response["criteria_met"]
                readable_explanation = f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
                readable_explanation_list.append(readable_explanation)
                rubric_items_with_grades.append({**rubric_item.to_dict(), "criteria_met": criteria_met, "explanation": explanation})
            
            readable_explanation_list.sort(key=lambda x: x.startswith("[False]"), reverse=True)
            sample["readable_explanation_str"] = "\n\n".join(readable_explanation_list)
            sample["rubric_items_with_grades"] = rubric_items_with_grades
            
            processed_samples.append(sample)

        final_metrics = _aggregate_get_clipped_mean(processed_samples)
        return final_metrics, processed_samples