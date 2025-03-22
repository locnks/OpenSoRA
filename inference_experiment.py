import os
import sys
import json

def get_bool(content):
    return True if content == "True" else False

def main():
    prompts = {"bird_fish": {"event_0": "Three birds are flying in the blue sky.", 
                             "event_1": "Some fish are swimming on the sand under the sea.", 
                             "event_0+event_1": "Three birds are flying in the blue sky. The scene gracefully transitions some fish are swimming on the sand under the water.",
                             "event_1+event_0": "Some fish are swimming on the sand under the water. The scene gracefully transitions to three birds are flying in the blue sky.",
                             "event_0~event_1": "Three birds are flying in the blue sky.  Some fish are swimming on the sand under the water."},
            #    "cat_dog": {"event_0": "A cat is sleeping in the room.", 
            #                "event_1": "A dog is running on the grass.", 
            #                "event_0+event_1": "A cat is sleeping in the room. The scene gracefully transitions to a dog is running on the grass.",
            #                "event_1+event_0": "A dog is running on the grass. The scene gracefully transitions to a cat is sleeping in the room.",
            #                "event_0~event_1": "A cat is sleeping in the room. A dog is running on the grass."},
            #    "person_station": {"event_0": "Several persons walks on the station.", 
            #                       "event_1": "A train passes through the station.", 
            #                       "event_0+event_1": "Several persons walks on the station. The scene gracefully transitions to a train passes through the station.",
            #                       "event_1+event_0": "A train passes through the station. The scene gracefully transitions to several persons walks on the station.",
            #                       "event_0~event_1": "Several persons walks on the station. A train passes through the station."},
            #    "rain_umbrella": {"event_0": "The rain falls heavily on the trees along the street.", 
            #                      "event_1": "The person opens an umbrella.", 
            #                      "event_0+event_1": "The rain falls heavily on the trees along the street. The scene gracefully transitions to the person opens an umbrella.",
            #                      "event_1+event_0": "The person opens an umbrella. The scene gracefully transitions to the rain falls heavily on the trees along the street.",
            #                      "event_0~event_1": "The rain falls heavily on the trees along the street. The person opens an umbrella."}
                                 }
    experiments = {
                "experiment-1_reversed": ["event_0", "event_1+event_0"], 
                "experiment0_reversed": ["event_1+event_0", "event_1"],
                "experiment1_reversed": ["event_1+event_0", "event_0"], 
                "experiment2_reversed": ["event_1", "event_0"],
                "experiment0": ["event_0+event_1", "event_1"],
                "experiment1": ["event_0+event_1", "event_0"], 
                "experiment2": ["event_0", "event_1"],
                "experiment-1": ["event_0", "event_0+event_1"], 
    }

    # experiments = {
    #                 "experiment-1": ["event_0", "event_0+event_1"], 
    #                 # "baseline": ["event_0+event_1"], 
    #                "experiment0": ["event_0+event_1", "event_1"],
    #                "experiment1": ["event_0+event_1", "event_0"], 
    #                 "experiment2": ["event_0", "event_1"]
    #                }
    # experiments = {
    #             "experiment-1_reversed": ["event_0", "event_1+event_0"], 
    #             "experiment0_reversed": ["event_1+event_0", "event_1"],
    #             "experiment1_reversed": ["event_1+event_0", "event_0"], 
    #             "experiment2_reversed": ["event_1", "event_0"],
    #         }
    # experiments = {"experiment_3": ["event_0~event_1", "event_0"],
    #               "experiment_4": ["event_1", "event_0~event_1"],
    #               "experiment_5": ["event_0~event_1", "event_1"]}
    experiment_configuration = "temporal,spatial,experiment\nFalse,False,True"
    with open("/home/stud/ghuang/Open-Sora/experiment_configuration", "w") as f:
        f.write(experiment_configuration)
    
    with open("/home/stud/ghuang/Open-Sora/experiment_configuration", "r") as f:
        content = f.readlines()
        conditions = content[1].split(',')
        temporal, spatial, experiment = get_bool(conditions[0]), get_bool(conditions[1]), get_bool(conditions[2])
        extension = ""
        middle = "_"
        if temporal:
            extension = "temporal"
            middle = "_block_level_fix_spatial_block_"
        elif spatial:
            extension = "spatial"
            middle = "_fix_spatial_block_"

    for key in prompts:
        for experiment in experiments:
            for ratio in range(0, 11):
                ratio = ratio / 10
                with open(f"/home/stud/ghuang/Open-Sora/{extension}_experiment_tmp", "w") as f:
                    f.write(f"{experiment}/" + str(ratio))
                filename = key + middle + experiment + "_" + str(ratio)
                
                candidates = []
                for element in experiments[experiment]:
                    candidates.append(prompts[key][element] + " aesthetic score: 6.5.")
                description = "-=-".join(candidates)

                print("Generating", filename, "with prompt", description)
        
                try:
                    os.system(f'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29200 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
            --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
            --prompt "{description}" --filename {filename}')
                except Exception:
                    print(f"Failed to generate video {filename}")

if __name__ == '__main__':
    main()