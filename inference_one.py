import os
import json

def main():
    description = "A car is driving on the road surrounding a mountain. There are a lot of dust behind the car. The car disappeared after a turn."
    filename = "guowen_test_no_use"

    try:
        os.system(f'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29100 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
--num-frames 32s --resolution 240p --aspect-ratio 9:16 \
--prompt "{description}" --filename {filename}')
    except Exception:
        print("Failed to generate video")

if __name__ == '__main__':
    main()