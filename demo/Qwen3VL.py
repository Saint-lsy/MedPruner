#  CUDA_VISIBLE_DEVICES=0 python -m tests.eval.Qwen3VL --methed medpruner

import argparse
import torch
from transformers import AutoProcessor, AutoConfig

import os,json,time
from qwen_vl_utils import process_vision_info

MODEL_PATH = "workspace/model/qwen3vl_8B"

TEMPERATURE=0
TOP_P=0.95
MAX_NEW_TOKENS=600
REPETITION_PENALTY=1.0
MAX_IMAGE_NUM=600

parser = argparse.ArgumentParser()
parser.add_argument(
  "--methed",
  default=os.environ.get("METHED", "orignal"),
  choices=["orignal", "medpruner"],
  help="Compression method. Can also be provided by env var METHED.",
)
args = parser.parse_args()

def _load_model_kwargs(methed):
  config_path = "config/config.json"
  with open(config_path, "r", encoding="utf-8") as f:
    config_data = json.load(f)
  if methed == "medpruner":
    print(f"run {methed}")
    from medpruner import Qwen3VLForConditionalGeneration
    return config_data.get(methed, {}), Qwen3VLForConditionalGeneration
  else:
    from transformers import Qwen3VLForConditionalGeneration
    return {},Qwen3VLForConditionalGeneration

# video_dir = "data/Eval/3DRad/valid_crop/valid_1000_a_1/"
# prompt = "<video>\nDoes this CT image present medical material?\nA.Yes\nB.No"
video_dir = "data/Eval/Amos/amos_test_crop_32/amos_5001/"
prompt = "<video>\nOffer a thorough analysis of the 3D image, leading to a list of findings and a proposed diagnosis."
# video_dir = "data/Eval/M3D/ct_quizze/008799/Axial_C__arterial_phase/"
# prompt = "<video>\nDoes this CT image present medical material?\nA.Yes\nB.No"
# video_dir = "data/Eval/Amos/amos_test_crop_32/amos_5071/"
# prompt = "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."

image_files = sorted([
  video_dir + f for f in os.listdir(video_dir)
  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.dcm'))
])

model_kwargs,model_class = _load_model_kwargs(args.methed)
model = model_class.from_pretrained(
  MODEL_PATH,
  trust_remote_code=True,
  torch_dtype=torch.bfloat16,
  device_map="cuda",
  attn_implementation="flash_attention_2",
  **model_kwargs
)
  
processor = AutoProcessor.from_pretrained(
  MODEL_PATH,
  trust_remote_code=True
)
tokenizer = processor.tokenizer
model.eval()

temperature = TEMPERATURE
top_p = TOP_P
max_new_tokens = MAX_NEW_TOKENS
repetition_penalty = REPETITION_PENALTY

messages = []
content = []
for i,image in enumerate(image_files):
  content.append({"type":"image","image":image})
content.append({"type":"text","text":prompt})
messages.append({"role":"user","content":content})

start_time = time.time()

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

do_sample = False if temperature == 0 else True
generated_ids = model.generate(**inputs,temperature=temperature,top_p=top_p,repetition_penalty=repetition_penalty,max_new_tokens=max_new_tokens,do_sample = do_sample)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
outputs = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
end_time = time.time()
print(f"time: {end_time - start_time:.2f}s")
print(f"result:{outputs}")
print(f"compression_rate:{model.model.compression_rate.float().cpu().item()}")
print(f"dyn_token_rate:{model.model.dyn_token_rate.float().cpu().item()}")
print(f"select_rate:{model.model.select_rate.float().cpu().item()}")
