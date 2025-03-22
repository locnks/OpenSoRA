import os
import json

def main():
    prompts = {"generalization_one": "A woman is dancing in a bright room.", "generalization_two": "A man is playing the piano in front of a window.", "generalization_three": "A girl is performing gymnastics in a stadium."}
    frames_list = {"8s": 102, "16s": 204, "32s": 408, "40s": 510, "52s": 663}
    for prompt_name in prompts:
        for seconds_number in frames_list:
            filename = f"{prompt_name}_{seconds_number}"
            description = prompts[prompt_name]
            try:
                os.system(f'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29400 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
        --num-frames {seconds_number} --resolution 720p --aspect-ratio 9:16 \
        --prompt "{description}" --filename {filename}')
            except Exception:
                print("Failed to generate video " + filename)

if __name__ == '__main__':
    main()