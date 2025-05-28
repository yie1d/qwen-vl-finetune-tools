import json
import math

from copy import deepcopy
from pathlib import Path

SYSTEM_PROMPT_1 = "基于给你的截图，我给你一个操作需求，你给我返回相关操作元素的bbox坐标，以JSON格式输出其bbox坐标"
SYSTEM_PROMPT_2 = "基于给你的截图，我给你一个操作需求，你给我返回相关操作元素的bbox坐标，以XML格式输出其坐标 <points x y>object</points>"


# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56 * 56,
                               max_pixels=14 * 14 * 4 * 1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)

    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))

    return [x1_new, y1_new, x2_new, y2_new]


def read_json(file_path: Path):
    with open(file_path, "r", encoding='utf8') as f:
        data = json.load(f)
    return data


def write_json(data, file_path: Path):
    with open(file_path, "w", encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    input_file = Path("../dataset/project-1-at-2025-05-28-10-06-a8720a12.json")
    output_file = Path("../dataset/mllm_rpa_action.json")
    input_data = read_json(input_file)

    output_data = []

    for item in input_data:
        for result_item in item['annotations'][0]['result']:
            if result_item['type'] != 'textarea':
                continue
            result_value = result_item['value']

            bbox = [
                result_value["x"] * result_item['original_width'] / 100,
                result_value["y"] * result_item['original_height'] / 100,
                (result_value['x'] + result_value["width"]) * result_item['original_width'] / 100,
                (result_value["y"] + result_value["height"]) * result_item['original_height'] / 100
            ]
            bbox = deepcopy(
                convert_to_qwen25vl_format(bbox, result_item['original_height'], result_item['original_width']))

            output_data.append({
                "messages": [
                    {
                        "value": SYSTEM_PROMPT_1,
                        "from": "system"
                    },
                    {
                        "value": f"<image>{result_value['text'][0]}",
                        "from": "human"
                    },
                    {
                        "value": f'{{\n"bbox_2d": {bbox}\n}}',
                        "from": "gpt"
                    }
                ],
                "images": [
                    f"/root/autodl-tmp/qwen-vl-finetune-tools/dataset/rpa_action/images/{item['file_upload'].split('-')[-1]}"
                ]
            })



    write_json(output_data, output_file)


if __name__ == '__main__':
    main()
