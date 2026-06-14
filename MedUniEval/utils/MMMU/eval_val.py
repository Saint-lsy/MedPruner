import torch
import os
import random

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets

from argparse import ArgumentParser
from mathruler.grader import extract_boxed_content
from .data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG,DOMAIN_CAT2SUB_CAT
from .eval_utils import evaluate,parse_multi_choice_response, parse_open_response
from ..utils import extract

def run_model(samples, model,):
    out_samples = []
    with torch.no_grad():
        messages_list = []
        current_messages = []
        current_samples = []
        for sample in tqdm(samples):
            messages = {"prompt":sample["final_input_prompt"],"image":sample["image"]}
            current_messages.append(messages)
            current_samples.append(sample)
            if len(current_messages) >= 2000:
                messages_list.append([current_messages,current_samples])
                current_messages = []
                current_samples = []
        if current_messages:
            messages_list.append([current_messages,current_samples])

        for current_messages,current_samples in tqdm(messages_list):
            outputs = model.generate_outputs(current_messages)
            for sample,response in zip(current_samples,outputs):
                out_sample = {
                    "id":sample['id'],
                    "question_type":sample['question_type'],
                    "answer":sample["answer"]
                }
                if "<answer>" in response:
                    response = extract(response,"answer")
                if extract_boxed_content(response) != "None":
                    response = extract_boxed_content(response)

                if sample['question_type'] == 'multiple-choice':
                    out_sample["all_choices"] = sample["all_choices"]
                    out_sample["index2ans"] = sample["index2ans"]
                    out_sample["response"] = response
                    out_sample["parsed_pred"] = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
                else:  # open question
                    out_sample["response"] = response
                    out_sample["parsed_pred"] = response
                out_samples.append(out_sample)
    return out_samples


def eval_MMMU_val(model,dataset_path,output_path,subset):
    total_results = {"total":{"total":0,"right":0}}
    for subject in tqdm(DOMAIN_CAT2SUB_CAT[subset]):
        sub_samples = []
        sub_dataset = load_dataset(dataset_path, subject, split="validation")

        eval_sub_path = os.path.join(output_path,subject)
        if not os.path.exists(eval_sub_path):
            os.mkdir(eval_sub_path)
        for sample in sub_dataset:
            sample = process_single_sample(sample)
            sample = construct_prompt(sample)
            sub_samples.append(sample)

        eval_samples = run_model(sub_samples, model)
        save_json(os.path.join(eval_sub_path,"output_sample.json"), eval_samples)
        judge_dict, metric_dict = evaluate(eval_samples)
        metric_dict.update({"num_example": len(eval_samples)})
        for eval_sample in eval_samples:
            eval_sample.update({"judge": judge_dict[eval_sample['id']]})

        save_json(os.path.join(eval_sub_path, 'parsed_output.json'), eval_samples)
        save_json(os.path.join(eval_sub_path, 'result.json'), metric_dict)
        total_results[subject] = metric_dict
        total_results["total"]["total"] += metric_dict["total"]
        total_results["total"]["right"] += metric_dict["right"]
    total_results["total"]["acc"] = total_results["total"]["right"] / total_results["total"]["total"]
    save_json(os.path.join(output_path, 'result.json'), total_results)
    return total_results
