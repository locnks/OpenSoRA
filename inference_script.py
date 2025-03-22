import os
import json

def main():
    prompt_file_name = "samples/samples/prompts.txt"
    count = []
    max_number = 10000
    with open(prompt_file_name, 'r') as f:
        prompts = f.readlines()
        for prompt in prompts:
            index, description = prompt.split(":")[0], prompt.split(":")[1]
            try:
                os.system(f'CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
    --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
    --prompt "{description}" --filename {str(index)}')
            except Exception:
                count.append(str(index))
                print('Failed to generate', description)
            max_number -= 1
            if max_number < 0:
                break
        print("Failed to generate", count)

if __name__ == '__main__':
    main()