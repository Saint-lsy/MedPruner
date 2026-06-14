
import os
import json
from PIL import Image
from tqdm import tqdm
from pydash import at
from mathruler.grader import extract_boxed_content
from ..question_formats import get_judgement_prompt,get_multiple_choice_prompt
from ..utils import save_json,extract,judge_multi_choice,judge_close_end_vqa,judge_judgement
from ..base_dataset import BaseDataset

class Radrestruct(BaseDataset):
    def __init__(self,model,dataset_path,output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx",0))
        self.num_chunks = int(os.environ.get("num_chunks",1))
    

    
    def load_data(self):
        dataset_path = self.dataset_path
        id_to_img_mapping_path = os.path.join(dataset_path,"id_to_img_mapping_frontal_reports.json")
        reports_path = os.listdir(os.path.join(dataset_path,"test_qa_pairs"))
        answer_options_path = os.path.join(dataset_path,"answer_options.json")


        dataset = []

        with open(id_to_img_mapping_path, "r") as f:
            id_to_img_mapping_frontal_reports = json.load(f)

        
        for report_file in reports_path:
            for elem in id_to_img_mapping_frontal_reports[report_file.split(".")[0]]:
                dataset.append((report_file,elem))

        with open(answer_options_path, "r") as f:
            answer_options = json.load(f)

        # ['index', 'Figure_path', 'Caption', 'Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D', 'Answer', 'split']
        for idx,sample in tqdm(enumerate(dataset)):
            if idx % self.num_chunks == self.chunk_idx:
                samples = self.construct_messages(sample,answer_options)
                self.samples.extend(samples)
        return self.samples

    def construct_messages(self,sample,answer_options):
        report,img = sample
        report_id = report.split(".")[0]
        img_path = os.path.join(self.dataset_path,"imgs",f"{img}.png")
        img = Image.open(img_path).convert("RGB")

        sample = {}
        samples = []
        with open(os.path.join(self.dataset_path,"test_qa_pairs",report), "r") as f:
            qa_pairs = json.load(f)
        
        for qa_pair_idx,qa_pair in enumerate(qa_pairs):
            question,answer,history,info = qa_pair
            options = info["options"]
            answer_idxs = at(answer_options,*options)
            answer_type = info["answer_type"]
        
            is_reasoning = True if os.environ.get("REASONING","False") == "True" else False
            if answer_type == "single_choice":
                prompt = get_judgement_prompt(question,is_reasoning)
            else:
                prompt = get_multiple_choice_prompt(question,options,is_reasoning)
            messages = {"prompt":prompt,"image":img}
            sample["messages"] = messages
            sample["answer_type"] = answer_type
            sample["question"] = question
            sample["choices"] = options
            sample["answer"] = answer[0]
            samples.append(sample)
        return samples


    def cal_metrics(self,out_samples):
        metrics = {
            "total metrics" : {
                "total":0,
                "right":0
            },
            "single_choice" : {
                "total" : 0,
                "right" : 0
            },
            "multiple_choice" : {
                "total" : 0,
                "right" : 0
            }
        }

        for i,out_sample in tqdm(enumerate(out_samples)):
            response = out_sample["response"]
            if extract_boxed_content(response)!= "None":
                response = extract_boxed_content(response)
            elif "<answer>" in response:
                response = extract(response,"answer")

            answer = out_sample["answer"]
            question = out_sample["question"]
            answer_type = out_sample["answer_type"]
            choices = out_sample["choices"]
            answer = answer.lower().strip()
            response = response.lower().strip()

            metrics["total metrics"]["total"] += 1
            if answer_type == "single_choice":
                metrics["single_choice"]["total"] += 1
                correct = judge_judgement(answer,response)
                out_samples[i]["correct"] = correct
                if correct:
                    metrics["single_choice"]["right"] += 1
                    metrics["total metrics"]["right"] += 1
            else:
                metrics["multiple_choice"]["total"] += 1
                correct = judge_multi_choice(choices,answer,response)
                out_samples[i]["correct"] = correct
                if correct:
                    metrics["multiple_choice"]["right"] += 1
                    metrics["total metrics"]["right"] += 1

        metrics["total metrics"]["acc"] = metrics["total metrics"]["right"]/metrics["total metrics"]["total"]
        metrics["single_choice"]["acc"] = metrics["single_choice"]["right"]/metrics["single_choice"]["total"]
        metrics["multiple_choice"]["acc"] = metrics["multiple_choice"]["right"]/metrics["multiple_choice"]["total"]

        return metrics,out_samples
