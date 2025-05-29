import io
import uuid
import ast
import torch
from PIL import Image
import threading
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# MODEL_PATH = "/root/autodl-tmp/ShowUI/model_weight/Qwen2.5-VL-7B-Instruct"
MODEL_PATH = "/root/autodl-tmp/qwen-vl-finetune-tools/output/qwen2_5vl_lora_sft"
MIN_PIXELS = 200704
MAX_PIXELS = 1053696
SYSTEM_PROMPT = "基于给你的截图，我给你一个操作需求，你给我返回相关操作元素的bbox坐标，以JSON格式输出其bbox坐标"


def singleton(cls):
    """
    装饰器，用于将类转换为单例模式。
    """
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance




@singleton
class Qwen2_5_VL_7B:
    _INIT = False
    
    def __init__(self):
        if self._INIT is False:
            self._INIT = True
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(
                MODEL_PATH,
                min_pixels=3136,
                max_pixels=12845056
            )
            self.system_prompt = SYSTEM_PROMPT

    def inference(self, img_data: io.BytesIO, prompt: str) -> list[int]:
        image = Image.open(img_data)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "image": uuid.uuid4().hex
                    }
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        input_height = inputs['image_grid_thw'][0][1] * 14
        input_width = inputs['image_grid_thw'][0][2] * 14

        bounding_boxes = self.parse_json(output_text[0])

        try:
            json_output = ast.literal_eval(bounding_boxes)
        except Exception:
            end_idx = bounding_boxes.rfind('"}') + len('"}')
            truncated_text = bounding_boxes[:end_idx] + "]"
            json_output = ast.literal_eval(truncated_text)

        if isinstance(json_output, list):
            bounding_box = json_output[0]
        else:
            bounding_box = json_output
        width, height = image.size
        return [
            int(bounding_box["bbox_2d"][0] / input_width * width),
            int(bounding_box["bbox_2d"][1] / input_height * height),
            int(bounding_box["bbox_2d"][2] / input_width * width),
            int(bounding_box["bbox_2d"][3] / input_height * height)
        ]

    @staticmethod
    def parse_json(json_output):
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i + 1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return json_output
