## inference commands
The inference speed of the following commands are:
```bash
# text to video
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall" --filename waterfall
```
to change the endpoint, use:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --rdzv-endpoint localhost:29400 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall" --filename waterfall
```

377 seconds worker-minor-2
275 seconds worker-5
1292 seconds worker-6 8s-video 4cards
```bash
# text to video
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nproc_per_node 6 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall"
```
NCCL timeout error
```bash
# text to video
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall" --filename waterfall
```
the cuda version is always 12.4 which doesn't work in opensora

The maxframes information is at `opensora/datasets/aspect.py`:
```python
NUM_FRAMES_MAP = {
    "1x": 51,
    "2x": 102,
    "4x": 204,
    "8x": 408,
    "16x": 816,
    "2s": 51,
    "4s": 102,
    "8s": 204,
    "16s": 408,
    "32s": 816,
}
```

64 failed and instead generated 5 videos, each is 2 seconds

## debugging
Set the environment variables for more information if encountering NCCL timeout error
```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_BLOCKING_WAIT=0  # not to enforce NCCL timeout
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
```
Every time the content of `opensora` is changed, run `pip install -v .` to apply the changes to the package.

## worker nodes information
- worker-1, 6 8000, CUDA 12.2, Version Error
- worker-2, 8 8000, CUDA 12.4, Version Error
- worker-3, 8 3090, CUDA 12.2, Memory Not Enough
- worker-4, 4 A100
- worker-5, 8 A6000, CUDA 12.2, NCCL Timeout Error, Solved
- worker-6, 8 A6000
- worker-7, 4 A40
- worker-8, 2 4090

## experiment questions and answers
> representation learning and directly predict the next frame in pixel -> a picture to demonstrate the example

> does one prompt will influence the next prompt, ie, will the random state be reset, or is it because of the training data
This is known as prompt ghosting and happens in stable diffusion. I used the following prompts to test:
first prompt: The sky is turning from red to purple.
second prompt: A group of black birds flying over the sky.
third prompt: A bird is standing on the tree.
[result](/home/stud/ghuang/Open-Sora/samples/random_state_reset_samples)

> whether the 8s videos have new problems than 4s videos or less problems

> use more detailed prompts to test the video generation performance. from opensora.utils.inference_utils import refine_prompts_by_openai
Examples are at [here](/home/stud/ghuang/Open-Sora/samples/refine_prompts_samples). The refine is using Refine.template to do manual refine.
- space_water
  Unrefined prompt: A cup of water is slowly poured out in the space station, releasing the liquid into the surrounding area
  Refined prompt: A close-up shot of a cup of water being slowly poured out in a zero-gravity environment inside a space station. As the liquid leaves the cup, it forms floating droplets that hover and move gently in the surrounding area. The background shows a metallic interior with panels and equipment typical of a space station, slightly out of focus. The lighting is soft and diffused, highlighting the clarity and surface tension of the water droplets. The scene is calm and focused on the movement of the liquid in microgravity. There are no additional objects or text visible in the video.
- stone_water
  Unrefined prompt: A piece of iron is gently placed on the surface of the water in a tank filled with water.
  Refined prompt: A close-up shot of a piece of iron being gently placed on the surface of water in a clear tank filled with water. The iron rests momentarily on the water's surface, creating small ripples and demonstrating the water's surface tension. The tank is transparent, allowing a clear view of the interaction between the iron and the water. The background is softly lit, with minimal distractions and no additional objects in the frame. The focus is on the piece of iron and the water's surface, emphasizing the delicate balance and physical interaction. There is no text or sound in the video.
- person_shoe
  Unrefined prompt: One person opens a bag, find shoes by the doorway and puts them inside, then closes the bag.
  Refined prompt: A close-up video of a person interacting with a bag in a tidy room. The person opens the bag, revealing an organized interior, and looks toward the doorway. Nearby, a pair of shoes is visible on the floor next to the door. The person picks up the shoes, carefully places them inside the bag, and zips it closed. The lighting is soft and natural, coming from a nearby window, creating a calm and casual atmosphere. The video focuses on the person's hands and movements, with the background slightly out of focus to emphasize the action.
- cat_room
  Unrefined prompt: The cat walks in the room, lick the milk in the bowl, and then jumps on the sofa.
  Refined prompt: A cozy living room with soft lighting and warm tones. A small cat enters through the doorway, moving gracefully toward a bowl of milk placed on the floor near a coffee table. The cat pauses, licks the milk in the bowl with delicate movements, and then looks around the room. It then leaps lightly onto a nearby sofa, curling up on a cushion. The focus of the video alternates between the cat's actions and the surrounding furniture, with the background slightly blurred to highlight the cat. The atmosphere is calm and inviting, with no additional objects or text distracting from the scene.

> why the generated video would have lego instead of real person

> the generation problem is  discussed partially here: https://github.com/hpcaitech/Open-Sora/issues/118

> prompt refinement can improve fidelity and details
As experimented [here](/home/stud/ghuang/Open-Sora/samples/refine_prompts_samples), the prompt refinement could add more details to the video.

> cat_room_refined the scene in the generated video changes suddenly, is it because of "The focus of the video alternates between the cat's actions and the surrounding furniture"? Remove the sentence or add the sentence to other prompts to see the effect
original prompt: A cozy living room with soft lighting and warm tones. A small cat enters through the doorway, moving gracefully toward a bowl of milk placed on the floor near a coffee table. The cat pauses, licks the milk in the bowl with delicate movements, and then looks around the room. It then leaps lightly onto a nearby sofa, curling up on a cushion. The focus of the video alternates between the cat's actions and the surrounding furniture, with the background slightly blurred to highlight the cat. The atmosphere is calm and inviting, with no additional objects or text distracting from the scene.
[video path](/home/stud/ghuang/Open-Sora/samples/refine_prompts_samples/sample_0000_cat_room_refined.mp4)
[video path](/home/stud/ghuang/Open-Sora/samples/samples/sample_0000_cat_room_1.mp4)
[video path](/home/stud/ghuang/Open-Sora/samples/samples/sample_0000_cat_room_2.mp4)

new prompt: A cozy living room with soft lighting and warm tones. A small cat enters through the doorway, moving gracefully toward a bowl of milk placed on the floor near a coffee table. The cat pauses, licks the milk in the bowl with delicate movements, and then looks around the room. It then leaps lightly onto a nearby sofa, curling up on a cushion. The atmosphere is calm and inviting, with no additional objects or text distracting from the scene.
[video path](/home/stud/ghuang/Open-Sora/samples/samples/sample_0000_cat_room_modified.mp4)

other experiments:
name: dog_room_modified
A medium shot of a dog walking out of a house into a sunny backyard with a grassy lawn. The dog, a medium-sized golden retriever, notices a brightly colored ball partially hidden in the grass and pauses briefly. The camera follows as the dog excitedly runs towards the ball, wagging its tail, and picks it up with its mouth. After a moment of play, the dog turns around and runs back towards the house with the ball. The lighting is natural, with soft shadows cast by nearby trees, and the focus remains on the dog throughout the scene. The background includes blurred elements of the yard, but no additional objects or people are present.
[video path](/home/stud/ghuang/Open-Sora/samples/scene_flash/sample_0000_dog_room_modified.mp4)

name: dog_room_unmodified
A medium shot of a dog walking out of a house into a sunny backyard with a grassy lawn. The dog, a medium-sized golden retriever, notices a brightly colored ball partially hidden in the grass and pauses briefly. The camera follows as the dog excitedly runs towards the ball, wagging its tail, and picks it up with its mouth. After a moment of play, the dog turns around and runs back towards the house with the ball. The lighting is natural, with soft shadows cast by nearby trees, and the focus alternates between the dog's actions and the surrounding environment,. The background includes blurred elements of the yard, but no additional objects or people are present.
[video path](/home/stud/ghuang/Open-Sora/samples/scene_flash/sample_0000_dog_room_unmodified.mp4)

name: bird_grass
category: unmodified
An aerial view of a bird flying gracefully over a dense, green forest, with the camera following its movements from a medium distance. The bird, a white seagull with outstretched wings, soars above the treetops, casting a faint shadow on the forest below. As it continues, the scene transitions smoothly to a shimmering blue lake reflecting the sky, with the bird gliding just above the water's surface. The journey proceeds as the bird flies over a sprawling grassland, where tall, golden grass sways gently in the wind. The lighting is soft and natural, with the sun low in the sky, creating warm tones and long shadows. The video captures a serene and uninterrupted sequence of the bird’s flight, with the landscape changing seamlessly beneath it.
[video path](/home/stud/ghuang/Open-Sora/samples/scene_flash/sample_0000_bird_grass.mp4)

name: basketball_hamburger
category: unmodified
A medium shot of a person playing basketball on an outdoor court during the day. The person, wearing a red jersey and black shorts, dribbles the ball skillfully before shooting it into a hoop. The surroundings include a clear blue sky, a few scattered trees, and the edges of a chain-link fence in the background, slightly out of focus. In the next scene, the same person is seated on a bench nearby, holding a hamburger with both hands and taking a bite. The lighting is natural, with sunlight casting soft shadows across the scene. The video seamlessly transitions between the active gameplay and the relaxed moment of eating, with no text or additional objects present.
[video path](/home/stud/ghuang/Open-Sora/samples/scene_flash/sample_0000_basketball_hamburger.mp4)

> explicity state the physics rule in the prompt to see the result -> failed
name: iron_water_sink
prompt: A close-up shot of a piece of iron being gently placed on the surface of water in a clear tank filled with water. The iron rests momentarily on the water's surface, creating small ripples and demonstrating the water's surface tension. The tank is transparent, allowing a clear view of the interaction between the iron and the water. The background is softly lit, with minimal distractions and no additional objects in the frame. The focus is on the process of iron sinking into the water. There is no text or sound in the video.
[video path](/home/stud/ghuang/Open-Sora/samples/explicit_physics_rule/sample_0000_iron_water_sink.mp4)

name: ballon_needle_burst
A close-up shot of a red balloon being pierced by a needle from left to right. The needle enters the balloon at a slight angle, causing the surface to stretch and form a small indentation around the point of contact. The balloon is fully inflated and appears glossy under bright, even lighting. The background is plain and out of focus, ensuring the focus remains on the balloon and needle. The motion is slow and detailed, capturing the process of ballon burst after being pierced by the needle. The video does not contain any text or additional objects.
[video path](/home/stud/ghuang/Open-Sora/samples/explicit_physics_rule/sample_0000_ballon_needle_burst.mp4)

name: eagle_lake
A close-up aerial view of an eagle soaring just above the surface of a tranquil lake, its powerful wings outstretched as it glides effortlessly. The eagle's feathers are sharply detailed, with hues of brown and white catching the soft sunlight. Below, the lake reflects the clear blue sky and surrounding trees, creating a shimmering mirror-like effect. Ripples form gently on the water as the bird's shadow dances across the surface. The background transitions between the lush greenery of distant forested shores and the open expanse of the lake. The video captures the serene and majestic flight of the eagle, with no additional objects or text present.
[video path](/home/stud/ghuang/Open-Sora/samples/explicit_physics_rule/sample_0000_eagle_lake.mp4)

name: water_freeze_ice
A close-up view of a transparent glass filled with clear water, placed on a flat surface in a cold environment. The water inside the glass is initially still, with tiny air bubbles visible near the surface. As the temperature drops, the water begins to freeze, with small ice crystals forming and expanding outward in intricate patterns. The process is gradual, with the transformation from liquid to solid captured in fine detail. The background is blurred, emphasizing the glass and the freezing water as the main focus. The lighting is soft and cool, highlighting the clarity and texture of the ice as it forms.
[video path](/home/stud/ghuang/Open-Sora/samples/explicit_physics_rule/sample_0000_water_freeze_ice.mp4)

name: red_blue_mix
A close-up view of two liquids, one vividly red and the other deep blue, being poured into a clear glass container from opposite sides. As they meet, the colors swirl and blend, creating dynamic patterns of purple in the mixing area. The movement of the liquids is smooth and continuous, with the distinct red and blue tones gradually merging into a vibrant gradient. The background is neutral and softly lit, ensuring full focus on the colors and their interaction. The lighting enhances the clarity of the liquid and highlights the blending process. There are no other objects or distractions in the video.
[video path](/home/stud/ghuang/Open-Sora/samples/explicit_physics_rule/sample_0000_red_blue_mix.mp4)

name: egg_rock
A slow-motion close-up of an egg falling and hitting a hard, dark gray rock surface. Upon impact, the shell cracks dramatically, sending small fragments outward while the yolk and egg white spill and spread unevenly across the rough texture of the rock. The lighting is natural, highlighting the glossy sheen of the yolk and the matte surface of the rock. The background is softly blurred, ensuring the focus remains on the egg and the moment of impact. The video captures the dynamic motion and fine details of the cracking process. There are no additional objects or text in the video.
[video path](/home/stud/ghuang/Open-Sora/samples/explicit_physics_rule/sample_0000_egg_rock.mp4)

name: stone_sink
A close-up shot of a smooth, gray stone sinking into clear water. The stone creates gentle ripples as it enters the surface, and small air bubbles form around it, rising toward the top. The water is calm, with a slight bluish tint, reflecting soft natural light. As the stone sinks, it becomes slightly blurred, emphasizing the depth and transparency of the water. The background consists of a subtle, out-of-focus aquatic environment with no additional objects. The video captures the motion of the stone and the interaction between the water and the light, with no text or additional distractions.
[video path](/home/stud/ghuang/Open-Sora/samples/explicit_physics_rule/sample_0000_stone_sink.mp4)

> run CogVideo5B, if not, run CogVideo2B to compare the result with OpenSora
The CogVideo5B model experiment is written in [Experiment.md](/home/stud/ghuang/CogVideo/Experiment.md).
