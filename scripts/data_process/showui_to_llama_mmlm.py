from copy import deepcopy
from pathlib import Path
from base import read_json, write_json, create_message, convert_to_qwen25vl_format, MAX_PIXELS, MIN_PIXELS, DATASET_PATH


def main():
    for DATASET_NAME in ['ShowUI-web', 'ShowUI-desktop']:
        input_file = Path(f"{DATASET_PATH}/{DATASET_NAME}/metadata/hf_train.json")
        output_file = Path(f"{DATASET_PATH}/mllm_{DATASET_NAME.lower()}.json")
        input_data = read_json(input_file)

        output_data = []

        for item in input_data:
            image_path = f"{DATASET_NAME}/images/{item['img_url'].removeprefix('/root/autodl-tmp/gui_database/ShowUI-web/images/')}"
            original_width, original_height = item['img_size']
            for result_item in item['element']:
                x1, y1, x2, y2 = result_item['bbox']

                bbox = [
                    x1 * original_width,
                    y1 * original_height,
                    x2 * original_width,
                    y2 * original_height
                ]
                bbox = deepcopy(convert_to_qwen25vl_format(
                    bbox,
                    original_height,
                    original_width,
                    min_pixels=MIN_PIXELS,
                    max_pixels=MAX_PIXELS
                ))

                output_data.append(create_message(
                    prompt=result_item['instruction'],
                    bbox=bbox,
                    image_path=image_path
                ))

        write_json(output_data, output_file)


if __name__ == '__main__':
    main()
