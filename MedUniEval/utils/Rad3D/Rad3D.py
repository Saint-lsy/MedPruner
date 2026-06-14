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
from ..eval_3d import evaluate_3drad
from ..mm_utils import load_images, process_images, load_video, process_video, tokenizer_multimodal_token, get_model_name_from_path, KeywordsStoppingCriteria
import concurrent.futures


class Rad3D(BaseDataset):
    def __init__(self, model, dataset_path, output_path):
        self.model = model
        self.output_path = output_path
        self.dataset_path = dataset_path if dataset_path else "./Rad3D_test.json"
        self.samples = []
        self.chunk_idx = int(os.environ.get("chunk_idx", 0))
        self.num_chunks = int(os.environ.get("num_chunks", 1))

    def load_data(self):
        json_path = self.dataset_path
        
        print(f"Loading 3D-RAD data from: {json_path}")
        
        with open(json_path + "/Rad3D_test.json", "r") as f:
            all_samples = json.load(f)
        
        print(f"Total samples in dataset: {len(all_samples)}")
        # Shuffle before chunking to avoid skewed per-chunk difficulty; deterministic via global seed.
        np.random.shuffle(all_samples)
        chunked_samples = np.array_split(all_samples, self.num_chunks)[self.chunk_idx].tolist()
        
        print(f"Chunk {self.chunk_idx}/{self.num_chunks}: Processing {len(chunked_samples)} samples")
        
        self.video_dirs = []
        for idx, sample in enumerate(tqdm(chunked_samples, desc=f"Loading chunk {self.chunk_idx}")):
            try:
                processed_sample,video_dir = self.construct_messages(sample)
                if processed_sample:
                    self.samples.append(processed_sample)
                    self.video_dirs.append(os.path.basename(video_dir))
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"Chunk {self.chunk_idx}: Successfully loaded {len(self.samples)} samples")
        return self.samples

    def construct_messages(self, sample):
        try:
            question = sample["conversations"][0]["value"]
            answer = sample["conversations"][1]["value"]
            question_type = sample.get("Question_Type", "open")
            main_type = sample.get("type", "Unknown")
            sub_type = sample.get("sub-type", "Unknown")
        except (KeyError, IndexError) as e:
            print(f"Warning: Invalid sample format. Error: {e}")
            return None
        
        video_dir = sample["video"][0]
        video_dir = os.path.join(self.dataset_path, video_dir)
        
        if not os.path.isdir(video_dir):
            print(f"Warning: Video directory not found: {video_dir}")
            return None

        images = []
        images, timestamps = load_video(video_dir, fps=1, max_frames=1800)
        
        if not images:
            print(f"Warning: No valid images loaded from: {video_dir}")
            return None
        
        question_clean = question.replace('<image>\n', '').replace('<video>\n', '').strip()
        
        if question_type.lower() in ['close', 'closed']:
            prompt = question_clean + "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = question_clean + "\nPlease answer the question concisely."
        
        messages = {
            "prompt": prompt,
            "images": images
        }
        sample["messages"] = messages
        
        return sample,video_dir

    def extract_choice_letter(self, text):
        text = str(text).lower().strip()
        
        match = re.match(r'^([a-d])[.\):]?\s*', text)
        if match:
            return match.group(1)
        
        if text in ['a', 'b', 'c', 'd']:
            return text
        
        return text

    def cal_metrics(self, out_samples):
        print("\n" + "="*80)
        print("Starting 3D-RAD Evaluation")
        print("="*80)
        
        TYPE_MAPPING = {
            "Medical_Computation": "Medical Computation",
            "Spatial_Relationship": "Spatial Relationship",
            "Abnormality_Detection": "Abnormality Detection",
            "Organ_Identification": "Organ Identification",
            "Image_Quality": "Image Quality"
        }
        
        SUB_TYPE_MAPPING = {
            "Thickness": "Thickness",
            "Distance": "Distance",
            "Volume": "Volume",
            "Density": "Density",
            "Location": "Location",
            "Relative_Position": "Relative Position",
            "Presence": "Presence",
            "Severity": "Severity",
            "Organ_Name": "Organ Name",
            "Organ_Count": "Organ Count",
            "Contrast": "Contrast",
            "Noise": "Noise"
        }
        
        results_text, metrics, wrong_answers = evaluate_3drad(out_samples)
        
        print(results_text)
        
        wrong_answers_readable = {}
        for key, value in wrong_answers.items():
            readable_key = TYPE_MAPPING.get(key, key)
            wrong_answers_readable[readable_key] = value
        
        os.makedirs(self.output_path, exist_ok=True)
        wrong_answers_path = os.path.join(self.output_path, "wrong_answers.json")
        
        with open(wrong_answers_path, 'w', encoding='utf-8') as f:
            json.dump(wrong_answers_readable, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Wrong answers saved to: {wrong_answers_path}")
        print("="*80 + "\n")
        
        return metrics, out_samples
