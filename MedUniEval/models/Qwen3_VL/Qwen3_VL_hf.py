import json
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor
from PIL import Image

def _load_model_kwargs(args):
    methed = args.compression
    config_path = args.config
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    if methed == "medpruner":
        print(f"run {methed}")
        from medpruner import Qwen3VLForConditionalGeneration
        return config_data.get(methed, {}), Qwen3VLForConditionalGeneration
    else:
        from transformers import Qwen3VLForConditionalGeneration
        return {},Qwen3VLForConditionalGeneration

class Qwen3VL:
    def __init__(self,model_path,args):
        super().__init__()
        model_kwargs,model_class = _load_model_kwargs(args)
        self.llm = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation ="flash_attention_2",
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.temperature = args.temperature
        self.top_p = args.top_p
        self.repetition_penalty = args.repetition_penalty
        self.max_new_tokens = args.max_new_tokens
        self.methed = args.compression

    def process_messages(self,messages):
        new_messages = []
        if "system" in messages:
            new_messages.append({"role":"system","content":messages["system"]}) 
        if "image" in messages:
            new_messages.append({"role":"user","content":[{"type":"image","image":messages["image"]},{"type":"text","text":messages["prompt"]}]})
        elif "images" in messages:
            content = []
            for i,image in enumerate(messages["images"]):
                content.append({"type":"image","image":image})
            content.append({"type":"text","text":messages["prompt"]})
            new_messages.append({"role":"user","content":content})
        else:
            new_messages.append({"role":"user","content":[{"type":"text","text":messages["prompt"]}]})
        messages = new_messages
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
)
        inputs = inputs.to("cuda")

        return inputs


    def generate_output(self,messages):
        inputs = self.process_messages(messages)
        do_sample = False if self.temperature == 0 else True
        generated_ids = self.llm.generate(**inputs,temperature=self.temperature,top_p=self.top_p,repetition_penalty=self.repetition_penalty,max_new_tokens=self.max_new_tokens,do_sample = do_sample)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
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
