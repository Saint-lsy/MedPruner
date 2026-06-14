from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os,torch
import concurrent.futures

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,6,7'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['OMP_NUM_THREADS'] = '8'  # Reduce CPU thread contention
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'  # Lazy load CUDA modules
class Qwen3VL:
    def __init__(self,model_path,args):
        super().__init__()
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            enforce_eager=True,
            trust_remote_code= True,
            limit_mm_per_prompt = {"image": args.max_image_num},
            seed=0,
            # enable_expert_parallel=True,
            # max_model_len=65536,  # M3D needs 65536 (66.82G), others 16384 (35G)
            # enable_chunked_prefill=True,
            # max_num_batched_tokens=4096,  # Used with scheduler chunked prefill
            disable_log_stats=True,  # Reduce logging overhead
            enable_prefix_caching=True,  # Enable prefix caching
            dtype="bfloat16",  # Explicit dtype
            # Disable unneeded features
            disable_custom_all_reduce=True,  # Not needed for PCIe GPUs
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens= args.max_new_tokens,
            stop_token_ids=[],
        )
        # self.qwen_max_image = 50

    def process_messages(self,messages):
        current_messages = []

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
                current_messages.append({"role":"system","content":system_prompt})
            if "image" in messages:
                image = messages["image"]
                current_messages.append({"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]})
            elif "images" in messages:
                content = []
                for i,image in enumerate(messages["images"]):
                    content.append({"type":"text","text":f"<image_{i+1}>: "})
                    content.append({"type":"image","image":image})
                content.append({"type":"text","text":messages["prompt"]})
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]}) 

        messages = current_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        # # Qwen3_VL specific (limit image count)
        # if len(image_inputs) > self.qwen_max_image:
        #     step = max(1, len(image_inputs) // self.qwen_max_image)
        #     image_inputs = image_inputs[::step]
        # # Final truncation to ensure strict limit
        # image_inputs = image_inputs[:self.qwen_max_image]
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt
        }
        if mm_data:
            llm_inputs["multi_modal_data"] = mm_data
        return llm_inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
    
    def generate_outputs(self,messages_list):
        # llm_inputs_list = [self.process_messages(messages) for messages in messages_list]
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            llm_inputs_list = list(executor.map(self.process_messages, messages_list))
        # from pdb import set_trace;set_trace()
        outputs = self.llm.generate(llm_inputs_list, sampling_params=self.sampling_params)
        res = []
        for output in outputs:
            generated_text = output.outputs[0].text
            res.append(generated_text)
        return res
