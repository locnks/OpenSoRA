import os
import json

def main():
    description = "An egg hits on a hard rock. -=- After a while, the scene transitions to a train is passing through the station."
    filename = "guowen_test_no_use"

    try:
        os.system(f'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29100 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
--num-frames 4s --resolution 720p --aspect-ratio 9:16 \
--prompt "{description}" --filename {filename}')
    except Exception:
        print("Failed to generate video")

if __name__ == '__main__':
    main()