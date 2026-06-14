from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import os,torch
from PIL import Image
import concurrent.futures

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '8'  # Reduce CPU thread contention
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # Lazy load CUDA modules

class MedGemma:
    def __init__(self,model_path,args):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            trust_remote_code= True,
            limit_mm_per_prompt = {"image": args.max_image_num},
            seed=0,
            enable_chunked_prefill=True,           # Enable chunked prefill
            max_num_batched_tokens=4096,           # Allow larger batching
            # enable_expert_parallel=True,
            # max_model_len=65536,  # M3D needs 65536 (66.82G), others 16384 (35G)
            disable_log_stats=True,  # Reduce logging overhead
            enable_prefix_caching=True,  # Enable prefix caching
            dtype="bfloat16",  # Explicit dtype
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            offload_buffers=True,
        )

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )
        # self.image_limit = 35

    def process_messages(self,messages):
        current_messages = []
        imgs = []
        images = messages["images"]
        if "messages" in messages:
            messages = messages["messages"]
            for message in messages:
                role = message["role"]
                content = message["content"]
                current_messages.append({"role":role,"content":[{"type":"text","text":content}]}) 

        else:
            prompt = messages["prompt"]
            if "system" in messages:
                system_prompt = messages["system"]
                current_messages.append({"role":"system","content":[{"type":"text","text":system_prompt}]})
            if "image" in messages:
                image = messages["image"]
                if isinstance(image,str):
                    image = Image.open(image)
                imgs.append(image)
                current_messages.append({"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]})
            elif "images" in messages:
                content = []
                content.append({"type":"text","text":messages["prompt"]})
                for i,image in enumerate(images,1):
                    content.append({"type": "image", "image": image})
                    content.append({"type": "text", "text": f"SLICE {i}"})
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]})
        
        messages = current_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # # MedGemma specific (limit image count)
        # if len(image_inputs) > self.image_limit:
        #     step = max(1, len(image_inputs) // self.image_limit)
        #     image_inputs = image_inputs[::step]
        # # Final truncation to ensure strict limit
        # image_inputs = image_inputs[:self.image_limit]

        vllm_input = {
            "prompt": prompt,
            "multi_modal_data": {"image": images}
        }
        
        return vllm_input


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_outputs(self,messages_list):
        llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     llm_inputs_list = list(executor.map(self.process_messages, messages_list))
        # from pdb import set_trace;set_trace()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res