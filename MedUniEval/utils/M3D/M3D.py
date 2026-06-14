import torch
import os
import json
import gc
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import re
from ..utils import save_json, extract, judge_multi_choice, judger, get_compare_messages, judge_open_end_vqa, judge_judgement
from ..base_dataset import BaseDataset
from ..question_formats import get_judgement_prompt, get_open_ended_prompt
from ..eval_3d import evaluate_m3d
from ..mm_utils import load_images, process_images, load_video, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
class M3D(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "./M3D_test.json"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))
        max_image_num_env = os.environ.get("max_image_num")
        self.max_image_num = int(max_image_num_env) if max_image_num_env else None

    def load_data(self):
        json_path = self.dataset_path
        
        print(f"Loading M3D data from: {json_path}")
        
        with open(json_path + "/M3D_test.json", "r") as f:
            all_samples = json.load(f)
        
        print(f"Total samples in dataset: {len(all_samples)}")
        
        chunked_samples = np.array_split(all_samples, self.num_chunks)[self.chunk_idx].tolist()
        
        print(f"Chunk {self.chunk_idx}/{self.num_chunks}: Processing {len(chunked_samples)} samples")
        
        for idx, sample in enumerate(tqdm(chunked_samples, desc=f"Loading chunk {self.chunk_idx}")):
            try:
                processed_sample = self.construct_messages(sample)
                if processed_sample:
                    self.samples.append(processed_sample)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"Chunk {self.chunk_idx}: Successfully loaded {len(self.samples)} samples")
        return self.samples

    def construct_messages(self, sample):
        try:
            question = sample["conversations"][0]["value"]
            answer = sample["conversations"][1]["value"]
            question_type = sample.get("Question_Type", "OPEN")
            Qtype = sample.get("type", "Default")
        except (KeyError, IndexError) as e:
            print(f"Warning: Invalid sample format. Error: {e}")
            return None
        
        video_dir = sample["video"][0]
        video_dir = os.path.join(self.dataset_path, video_dir)
        images = []
        question_clean = question.replace('<image>\n', '').replace('<video>\n', '').strip()
        
        if question_type.upper() in ['CLOSE', 'CLOSED']:
            prompt = question_clean + "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = question_clean + "\nPlease answer the question concisely."
        
        messages = {
            "prompt": prompt,
            "images": self._sample_frames(video_dir)
        }
        sample["messages"] = messages
        
        return sample

    def _sample_frames(self, video_dir):
        if not self.max_image_num or self.max_image_num <= 0:
            return video_dir

        if not os.path.isdir(video_dir):
            return video_dir

        frame_files = [f for f in sorted(os.listdir(video_dir)) if os.path.isfile(os.path.join(video_dir, f))]
        if len(frame_files) <= self.max_image_num:
            return [os.path.join(video_dir, f) for f in frame_files]

        indices = np.linspace(0, len(frame_files) - 1, self.max_image_num, dtype=int)
        sampled_files = [frame_files[i] for i in indices]
        return [os.path.join(video_dir, f) for f in sampled_files]

    def extract_choice_letter(text):
    
        text = str(text).lower().strip()
        
        match = re.match(r'^([a-d])[.\):]?\s*', text)
        if match:
            return match.group(1)
        
        if text in ['a', 'b', 'c', 'd']:
            return text
        
        return text

    def cal_metrics(self, out_samples):
        
        print("\n" + "="*80)
        print("Starting M3D Evaluation")
        print("="*80)
        QUESTION_TYPE_MAPPING = {
            1: "Plane", 2: "Phase", 3: "Organ", 4: "Abnormality", 5: "Location"
}
        results_text, metrics, wrong_answers = evaluate_m3d(out_samples)
        
        print(results_text)
        
        wrong_answers_readable = {
            QUESTION_TYPE_MAPPING.get(k, f"Type_{k}"): v 
            for k, v in wrong_answers.items()
        }
        
        os.makedirs(self.output_path, exist_ok=True)
        wrong_answers_path = os.path.join(self.output_path, "wrong_answers.json")
        
        with open(wrong_answers_path, 'w', encoding='utf-8') as f:
            json.dump(wrong_answers_readable, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Wrong answers saved to: {wrong_answers_path}")
        print("="*80 + "\n")
        
        return metrics, out_samples