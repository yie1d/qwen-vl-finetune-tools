from copy import deepcopy
from pathlib import Path
from base import read_json, write_json, create_message, convert_to_qwen25vl_format, MAX_PIXELS, MIN_PIXELS, DATASET_PATH

DATASET_NAME = 'rpa_action'


def main():
    input_file = Path(f"{DATASET_PATH}/{DATASET_NAME}/project-1-at-2025-05-28-10-06-a8720a12.json")
    output_file = Path(f"{DATASET_PATH}/mllm_{DATASET_NAME.lower()}.json")
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
            bbox = deepcopy(convert_to_qwen25vl_format(
                bbox,
                result_item['original_height'],
                result_item['original_width'],
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS
            ))

            output_data.append(create_message(
                prompt=result_value['text'][0],
                bbox=bbox,
                image_path=f"{DATASET_NAME}/images/{item['file_upload'].split('-')[-1]}"
            ))

    write_json(output_data, output_file)


if __name__ == '__main__':
    main()
