import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm
import os
import concurrent.futures

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '8'  # Reduce CPU thread contention
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # Lazy load CUDA modules

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

class HuluMed_Qwen2:
    def __init__(self, model_path, args):
        super().__init__()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(), # Number of parallel tasks; if GPUs < tasks, multi-GPU for one model
            enforce_eager=True,
            trust_remote_code= True,
            limit_mm_per_prompt = {"image": args.max_image_num},
            seed=0,
            # gpu_memory_utilization=0.85,
            disable_log_stats=True,  # Reduce logging overhead
            enable_prefix_caching=True,  # Enable prefix caching
            # Disable unneeded features
            disable_custom_all_reduce=True,  # Not needed for PCIe GPUs
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )

    def process_messages(self, messages):
       
        prompt = messages.get("prompt", "")
        
        conversation = [{"role": "user", "content": ""}]
        
        loaded_images = None

        image_paths_or_pil = messages.get("images") or ([messages["image"]] if "image" in messages else [])
        if image_paths_or_pil:
            loaded_images = load_images(image_paths_or_pil)
            conversation[0]["content"] = conversation[0]["content"] + "<image>"*len(loaded_images)
        conversation[0]["content"] = conversation[0]["content"] + prompt
        
        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        vllm_input = {
          "prompt": prompt,
          "multi_modal_data": {"image": loaded_images}
        }
      
            
        return vllm_input


    def generate_output(self, messages):

        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_outputs(self,messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        # from pdb import set_trace;set_trace()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res