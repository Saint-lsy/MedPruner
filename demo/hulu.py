#  CUDA_VISIBLE_DEVICES=0 python -m tests.eval.hulu --methed medpruner

import argparse
import torch
from transformers import AutoProcessor
from PIL import Image
import os,json

MODEL_PATH="workspace/model/hulu_7B"

TEMPERATURE=0
TOP_P=0.95
MAX_NEW_TOKENS=1500
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
        from medpruner import HulumedQwen2ForCausalLM
        return config_data.get(methed, {}),HulumedQwen2ForCausalLM

def load_images(image_path):
        images = []
        def safe_open(f):
            try:
                with Image.open(f).convert('RGB') as img:
                    return img
            except Exception:
                pass  

        if isinstance(image_path, str) and os.path.isfile(image_path):
            img = safe_open(image_path)
            if img is not None:
                images.append(img)

        elif isinstance(image_path, str) and os.path.isdir(image_path):
            for f in sorted(os.listdir(image_path)):
                full_path = os.path.join(image_path, f)
                if os.path.isfile(full_path):
                    img = safe_open(full_path)
                    if img is not None:
                        images.append(img)

        elif isinstance(image_path, list) and isinstance(image_path[0], str):
            for f in image_path:
                img = safe_open(f)
                if img is not None:
                    images.append(img)

        elif isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
            images = [img.convert('RGB') for img in image_path]

        elif isinstance(image_path, Image.Image):
            images = [image_path.convert('RGB')]

        else:
            raise ValueError(f"Unsupported image path type: {type(image_path)}")

        return images

# video_dir = "data/Eval/3DRad/valid_crop/valid_1000_a_1/"
# prompt = "<video>\nDoes this CT image present medical material?\nA.Yes\nB.No"
video_dir = "data/Eval/Amos/amos_test_crop_32/amos_0034/"
prompt = "You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. Return the report in the following format: Findings: {} Impression: {}."

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

conversation = [{"role": "user", "content": []}]
loaded_images = load_images(image_files)
if len(loaded_images) > 5:
  conversation[0]["content"].append({"type": "video", "num_frames": len(loaded_images)})
elif 0 < len(loaded_images) <= 5:
  for _ in loaded_images:
    conversation[0]["content"].append({"type": "image"})
conversation[0]["content"].append({"type": "text", "text":prompt })
inputs = processor(
    images=[loaded_images] if loaded_images is not None else None,
    conversation=conversation,
    add_system_prompt=False,
    add_generation_prompt=True,
    return_tensors="pt"
)

inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs and inputs["pixel_values"] is not None:
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

print(f"{'='*10}start generate{'='*10}")
do_sample = True if temperature > 0 else False
  
with torch.inference_mode():
  output_ids = model.generate(
    **inputs,
    do_sample=False,
    temperature=temperature if do_sample else 0,
    #top_p=top_p,
    repetition_penalty = repetition_penalty,
    max_new_tokens=max_new_tokens,
    use_cache=True,
    pad_token_id=tokenizer.eos_token_id,
  )

outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

print(f"result:{outputs}")
print(f"compression_rate:{model.compression_rate.float().cpu().item()}")
print(f"dyn_token_rate:{model.dyn_token_rate.float().cpu().item()}")
print(f"select_rate:{model.select_rate.float().cpu().item()}")

