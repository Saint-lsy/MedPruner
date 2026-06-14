from transformers import AutoProcessor
import os,json
import torch

from tqdm import tqdm

from PIL import Image

def _load_model_kwargs(args):
    methed = args.compression
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    if methed == "medpruner":
        print(f"run {methed}")
        from medpruner import Gemma3ForConditionalGeneration
        return config_data.get(methed, {}), Gemma3ForConditionalGeneration
    else:
        from transformers import AutoModelForImageTextToText
        return {},AutoModelForImageTextToText

class MedGemma:
    def __init__(self,model_path,args):
        model_kwargs,model_class = _load_model_kwargs(args)
        super().__init__()
        self.llm = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation ="flash_attention_2",
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        self.methed = args.compression

    def process_messages(self,messages):
        current_messages = []
        imgs = []
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
                for i,image in enumerate(messages["images"]):
                    if isinstance(image,str):
                        image = Image.open(image)
                    content.append({"type":"text","text":f"SLICE {i+1}: "})
                    content.append({"type":"image","image":image})
                    imgs.append(image)
                content.append({"type":"text","text":messages["prompt"]})
                current_messages.append({"role":"user","content":content})
            else:
                current_messages.append({"role":"user","content":[{"type":"text","text":prompt}]}) 
        
        
        inputs = self.processor.apply_chat_template(
            current_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.llm.device, dtype=torch.bfloat16)

        return inputs


    def generate_output(self,messages):
        llm_inputs = self.process_messages(messages)
        input_len = llm_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            do_sample = False if self.temperature == 0 else True
            generation = self.llm.generate(**llm_inputs,max_new_tokens=self.max_new_tokens,do_sample = do_sample, pad_token_id= self.processor.tokenizer.eos_token_id,top_k = None,top_p = None)
            generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        print(decoded)
        return decoded
    
    def generate_outputs(self,messages_list):
        res = []
        compose_rates = []
        for messages in tqdm(messages_list):
            result = self.generate_output(messages)
            res.append(result)
            if self.methed == "medpruner":
                compose_rates.append({
                    "compression_rate" : self.llm.model.compression_rate.float().cpu().item(),
                    "dyn_token_rate" : self.llm.model.dyn_token_rate.float().cpu().item(),
                    "select_rate" : self.llm.model.select_rate.float().cpu().item()
                })
        return res,compose_rates
