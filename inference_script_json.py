import os
import json

def main():
    prompt_file_name = 'prompts.json'
    count = []
    max_number = 10000
    with open(prompt_file_name, 'r') as f:
        prompts = json.load(f)
        for prompt in prompts:
            try:
                os.system(f'CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
    --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
    --prompt "{prompt["prompts"]}" --filename {str(prompt["index"]) + "_" + prompt["category"]}')
            except Exception:
                count.append(prompt["index"])
                print('Failed to generate', prompt)
            max_number -= 1
            if max_number < 0:
                break
        print("Failed to generate", count)

if __name__ == '__main__':
    main()