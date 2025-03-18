import os
import json

def main():
    description = "A cat is sleeping in the room. -=- After a while, the scene transitions to a dog is running on the grassland."
    # description = "An egg hits on a hard rock. -=- After a while, the scene transitions to birds are flying in the sky."
    # description = "After a while, the scene transitions to birds are flying in the sky."


    prompts = {
                "cat_dog_room_grass": "A cat is sleeping in the room. -=- After a while, the scene transitions to a dog is running on the grassland.",
               "light_night_star": "The light is shining on the table. -=- The scene then transitions to stars shining in the dark night.",
               "egg_rock_dog": "An egg hits on a hard rock. -=- The scene then transitions to a dog is running on the grassland.",
               "egg_rock_station": "An egg hits on a hard rock. -=- After a while, the scene transitions to a train is passing through the station.",
               "bird_lake_tree": "A bird is flying over a lake. -=- After a while, the bird lands on a tree.",
               "egg_rock_bird": "An egg hits on a hard rock. -=- After a while, the scene becomes several birds flying in the forest."}
    mark_index = {
        "cat_dog_room_grass": 10,
        "light_night_star": 9,
        "egg_rock_dog": 10,
        "egg_rock_station": 10,
        "bird_lake_tree": 11,
        "egg_rock_bird": 10
    }
    experiments = [
                    # "_causal_mask_experiment_0_", 
                   "_causal_mask_experiment_1_"
                   ]

    for key in prompts:
        for exp in experiments:
            for i in range(0, 20):
                ratio = round(i / 20, 2)
                filename = key + exp + str(ratio)
                print("Generating ", filename)
                with open("/home/stud/ghuang/Open-Sora/causal_mask_ratio", "w") as f:
                    f.write(key + "-=-" + str(mark_index[key]) + "-=-" + exp + "-=-" + str(ratio))
                try:
                    os.system(f'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29200 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
            --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
            --prompt "{prompts[key]}" --filename {filename}')
                except Exception:
                    print("Failed to generate video")

if __name__ == '__main__':
    main()