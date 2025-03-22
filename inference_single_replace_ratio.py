import os
import json

def main():
    # description = "Two lions are running fast on the African grassland under the sunset."
    # descriptions_list = {"smoke_grass": "A man is smoking on the balcony. He sees kids playing on the grassland.",
    #                      "lions_grassland": "Two lions are running fast on the African grassland under the sunset.",
    #                      "tai_chi": "An elderly practitioner performs Tai Chi in a misty garden.", 
    #                      "first_step": "A toddler takes her first unaided step slowly.", 
    #                      "astronaut_space": "An astronaut reaches for a drifting tool and slowly walks in the space."}
    descriptions_list = {"brown_egg_rock": "A brown egg hits on a hard rock."}

    orders = ["last"]
    ratios = ["1.0"]

    for key in descriptions_list:
        for order in orders:
            for ratio in ratios:
                with open("/home/stud/ghuang/Open-Sora/tmp", "w") as f:
                    f.write(order + "-" + ratio)
                filename = key
                description = descriptions_list[key]
                filename += "_" + order + "_" + ratio
        
                try:
                    os.system(f'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --rdzv-endpoint localhost:29300 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
            --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
            --prompt "{description}" --filename {filename}')
                except Exception:
                    print("Failed to generate video")

if __name__ == '__main__':
    main()