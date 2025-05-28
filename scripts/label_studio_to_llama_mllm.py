import json
from copy import deepcopy
from pathlib import Path

SYSTEM_PROMPT_1 = "基于给你的截图，我给你一个操作需求，你给我返回相关操作元素的bbox坐标，以JSON格式输出其bbox坐标"
SYSTEM_PROMPT_2 = "基于给你的截图，我给你一个操作需求，你给我返回相关操作元素的bbox坐标，以XML格式输出其坐标 <points x y>object</points>"


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

            _data = {
                "messages": [
                    {
                        "value": '',
                        "from": "system"
                    },
                    {
                        "value": f"<image>{result_value['text'][0]}",
                        "from": "human"
                    },
                    {
                        "value": f'{{\n"bbox_2d": ['
                                 f'{int(result_value["x"])}, '
                                 f'{int(result_value["y"])}, '
                                 f'{int(result_value["x"] + result_value["width"])}, '
                                 f'{int(result_value["y"] + result_value["height"])}'
                                 f']\n}}',
                        "from": "gpt"
                    }
                ],
                "images": [
                    f"/root/autodl-tmp/dataset/rpa_action/images/{item['file_upload'].split('-')[-1]}"
                ]
            }

            _data['messages'][0]['value'] = SYSTEM_PROMPT_1
            _data['messages'][2]['value'] = (f'{{"bbox_2d": ['
                                             f'{int(result_value["x"])}, '
                                             f'{int(result_value["y"])}, '
                                             f'{int(result_value["x"] + result_value["width"])}, '
                                             f'{int(result_value["y"] + result_value["height"])}]}}')
            output_data.append(deepcopy(_data))
            _data['messages'][0]['value'] = SYSTEM_PROMPT_2
            _data['messages'][2]['value'] = (f'<points x1="{int(result_value["x"] + result_value["width"] / 2)}" '
                                             f'y1="{int(result_value["y"] + result_value["height"] / 2)}" '
                                             f'alt="{result_value["text"][0]}">{result_value["text"][0]}</points>')
            output_data.append(deepcopy(_data))

    write_json(output_data, output_file)


if __name__ == '__main__':
    main()
