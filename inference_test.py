import os
import json

def main():
    description = "A cat is sleeping in the room. -=- After a while, the scene transitions to a dog is running on the grassland."
    # description = "An egg hits on a hard rock. -=- After a while, the scene transitions to birds are flying in the sky."
    # description = "After a while, the scene transitions to birds are flying in the sky."

    for i in range(0, 20):
        ratio = round(i / 20, 2)
        filename = "cat_dog_room_grass"
        filename += f"_causal_mask_experiment_0_{str(ratio)}"
        with open("/home/stud/ghuang/Open-Sora/causal_mask_ratio", "w") as f:
            f.write(str(ratio))
        try:
            os.system(f'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29200 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
    --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
    --prompt "{description}" --filename {filename}')
        except Exception:
            print("Failed to generate video")

if __name__ == '__main__':
    main()