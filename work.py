import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import seaborn as sns
import torch
from PIL import Image
import io
from tqdm import tqdm
from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import MultipleLocator

# attention_map_path = "/home/stud/ghuang/Open-Sora/samples/samples/attention_maps_bird_flying_no_attn_bias"
# attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_cats_grass/"
# attention_map_path = "/home/stud/ghuang/Open-Sora/samples/samples/attention_maps_girl_bike_no_attn_bias"
# attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_girl_bike_50_denoising_steps_4s"
# attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_girl_bike_100_denoising_steps_4s"
# attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_girl_bike_20_denoising_steps_4s"
category = "egg_rock_2s"
# attention_map_path = "/home/stud/ghuang/Open-Sora/samples/samples/attention_maps_scene_flash_two"
# attention_map_path = "/home/stud/ghuang/Open-Sora/samples/samples/attention_maps_scene_flash_three"
# attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_egg_rock_2s"
# attention_map_path = "/home/stud/ghuang/Open-Sora/samples/samples/attention_maps_scene_flash_one"
# attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_bird_lake"
# attention_map_path = "/home/stud/ghuang/Open-Sora/samples/samples/attention_maps_boy_beach"
attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_egg_rock/"
# attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_dog_mirror"
# attention_map_path = "/home/stud/ghuang/Open-Sora/samples/samples/attn_maps_girl_bike_align_7"
# attention_map_path = "/mnt/data8/liao/ghuang/visualization/attn_maps_girl_bike_50_denoising_steps" # this is 8s video
# attention_map_path = "/home/stud/ghuang/Open-Sora/samples/samples/attn_maps_girl_bike_align_condition_7"

# TODO: whether the '_' is used as the padding token
# scene_flash_one
# tokens = ['▁', 'a', '▁bird', '▁', 'f', 'lies', '▁in', '▁the', '▁sky', '.', '▁the', '▁scene', '▁then', '▁transition', 'e', 'd', '▁to', '▁', 'a', '▁ball', '▁rolling', '▁on', '▁the', '▁grass', 'l', 'and', '.', '▁then', '▁the', '▁scene', '▁transition', 'e', 'd', '▁to', '▁', 'a', '▁cat', '▁lying', '▁in', '▁the', '▁room', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
# scene_flash_two
# tokens = ['▁', 'a', '▁man', '▁is', '▁sitting', '▁in', '▁the', '▁room', '.', '▁the', '▁scene', '▁then', '▁becomes', '▁', 'a', '▁squirrel', '▁running', '▁on', '▁the', '▁tree', '.', '▁then', '▁the', '▁scene', '▁flash', 'e', 'd', '▁into', '▁fire', '▁floating', '▁on', '▁the', '▁water', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>',]

# tokens = ['▁the', '▁bird', '▁is', '▁flying', '▁above', '▁', 'a', '▁lake', '.', '▁and', '▁then', '▁it', '▁', 'f', 'lies', '▁over', '▁', 'a', '▁green', '▁forest', '.', '▁and', '▁then', '▁it', '▁passes', '▁through', '▁', 'a', '▁golden', '▁grass', 'l', 'and', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
# tokens = ['▁', 'a', '▁bird', '▁', 'f', 'lies', '▁in', '▁the', '▁sky', '.', '▁the', '▁scene', '▁then', '▁transition', 'e', 'd', '▁to', '▁', 'a', '▁ball', '▁rolling', '▁on', '▁the', '▁grass', 'l', 'and', '.', '▁then', '▁the', '▁scene', '▁transition', 'e', 'd', '▁to', '▁', 'a', '▁cat', '▁lying', '▁in', '▁the', '▁room', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
# tokens = ['▁', 'a', '▁girl', '▁is', '▁riding', '▁', 'a', '▁bike', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
# tokens = ['▁', 'a', '▁bird', '▁is', '▁flying', '▁over', '▁', 'a', '▁lake', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
# tokens = ['▁', 'a', '▁boy', '▁is', '▁running', '▁on', '▁the', '▁beach', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
# tokens = ['▁three', '▁white', '▁cats', '▁runs', '▁on', '▁the', '▁grass', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
tokens = ['▁', 'a', '▁brown', '▁egg', '▁hits', '▁on', '▁', 'a', '▁hard', '▁rock', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
# tokens = ['▁the', '▁dog', '▁is', '▁sitting', '▁in', '▁front', '▁of', '▁', 'a', '▁mirror', '▁and', '▁then', '▁lies', '▁down', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']
# tokens = ['▁', 'a', '▁group', '▁of', '▁birds', '▁is', '▁flying', '▁in', '▁the', '▁sky', '.', '▁the', '▁the', '▁scene', '▁transition', 'e', 'd', '▁to', '▁the', '▁fish', 'e', 's', '▁underneath', '▁the', '▁sea', '.', '▁then', '▁the', '▁scene', '▁transition', 'e', 'd', '▁to', '▁kids', '▁running', '▁on', '▁the', '▁playground', '.', '▁aesthetic', '▁score', ':', '▁', '6.5', '.', '</s>']

def create_gif_matplotlib(image_dir, output_path, interval=500):
    image_files = []
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    for ext in valid_extensions:
        image_files.extend(sorted(Path(image_dir).glob(f'*{ext}')))
    
    if not image_files:
        raise ValueError("No valid images found in the specified directory")
    fig = plt.figure()
    plt.axis('off')  # Hide axes
    images = []
    for filename in image_files:
        try:
            img = mpimg.imread(filename)
            im = plt.imshow(img, animated=True)
            images.append([im])
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
    ani = animation.ArtistAnimation(fig, images, interval=interval, 
                                  blit=True, repeat_delay=1000)
    ani.save(output_path, writer='pillow')
    plt.close()
    print(f"GIF created successfully at: {output_path}")


def create_gif_denoising_process(tensor_list, prefix, filename_suffix):
    height, width = tensor_list[0].shape
    fig, ax = plt.subplots(figsize=(16, 9))
    cax = ax.imshow(tensor_list[0], cmap="viridis", interpolation="nearest")
    plt.colorbar(cax, ax=ax, orientation="vertical")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.set_title(f"{prefix}cross attention map - level 1")
    ax.set_xlabel(f"token dimension, total={width}")
    ax.set_ylabel(f"temporal dimension, total={height}")
    
    def update(tensor_index):
        cax.set_array(tensor_list[tensor_index])
        ax.set_title(f"{prefix}cross attention map - block_no {tensor_index + 1}/28")
    ani = FuncAnimation(fig, update, frames=len(tensor_list), interval=500)
    ani.save(f"/home/stud/ghuang/Open-Sora/z/visualization_{filename_suffix}.gif", writer=PillowWriter(fps=2))

def create_array_gif(arrays, output_filename='animation.gif', duration=300):
    frames = []
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    
    index = 0
    for array in tqdm(arrays):
        ax.clear()
        
        ax.plot(array, '-o')
        ax.set_title(f'hidden_dimension_index={index}')
        index += 1
        ax.set_xlabel('length(first 160), total=45x80=3600')
        ax.set_ylabel('value')
        
        # ax.set_xlim(0, 200)
        # ax.set_ylim(0, 35)
        ax.grid(True)

        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        buf.close()
    
    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    plt.close()

filenames = os.listdir(attention_map_path)
sorted_filenames = sorted(filenames)

filename_suffix = category

attention_map_list = []
token_length = 2 * len(tokens)

# =====>
# prefix = "timestep_embedding"
# filename_suffix = "timestep_embedding"

# attention_map = torch.load("/home/stud/ghuang/Open-Sora/z_positional_embedding19-20-37-163727.pt", map_location=torch.device('cpu')).float()
# attention_map = attention_map.reshape(45, 80, 1152)
# attention_map = attention_map.reshape(3600, 1152)[:160]
# tensor_list = []
# for i in range(1152):
#     tensor_list.append(attention_map[..., i])
# create_array_gif(tensor_list)

# for hidden_dimension_index in range(1152):
#     attention_map = torch.load("/home/stud/ghuang/Open-Sora/z_positional_embedding19-20-37-163727.pt", map_location=torch.device('cpu')).float()
#     attention_map = attention_map.reshape(45, 80, 1152)
#     attention_map = attention_map[..., hidden_dimension_index]
#     attention_map = attention_map.reshape(3600)[:100]
#     averages = attention_map
#     x = np.arange(len(averages))
#     plt.figure()
#     plt.plot(x, averages.numpy(), '-o')
#     # plt.xticks(ticks=x)
#     plt.title(f"z_positional_embedding visualization, hidden_dimension_index={hidden_dimension_index}")
#     plt.xlabel("length(first 160), total=3600=45x80")
#     plt.ylabel("value")
#     plt.grid(True)
#     plt.savefig("/home/stud/ghuang/Open-Sora/z/" + filename_suffix + "_" + str(hidden_dimension_index) + "_blink.png", dpi=300, bbox_inches="tight")
#     plt.close()


# block_level_attention
# block_index = 2
# start_number = block_index * 2# notice that a level have two blocks, so 52 means spatial_block_index=26 

# start_number = 1624
# end_number = 6000
# step = 2
# token_index = 0

# attention_list = []

# for token_index in range(len(tokens)):
#     for i, name in tqdm(enumerate(sorted_filenames[start_number:end_number:step])): # 28 depths, each level has one temporal block and one spatial block.
#         attention_map = torch.load(os.path.join(attention_map_path, name), map_location=torch.device('cpu')).float()

#         # attention_map = attention_map[..., :attention_map.shape[2] // 2, :token_length // 2]
#         # attention_map = attention_map.reshape(1, 16, 30, 45, 80, token_length // 2)
#         # attention_map = attention_map[0]
#         # attention_map = attention_map.mean(dim=0)

#         attention_map = attention_map[:attention_map.shape[0] // 2, :token_length // 2]
#         attention_map = attention_map.reshape(30, 45, 80, token_length // 2)

#         attention_map = attention_map[..., token_index]

#         # attention_map_list.append(attention_map.mean(dim=0))
#         for tensor_index in range(30):
#             attention_map_list.append(attention_map[tensor_index])
#         # create_gif_denoising_process(attention_map_list, prefix, filename_suffix + "_" + str(token_index) + "_" + str(i))

#         averages = attention_map.mean(dim=(0, 1, 2))
#         attention_list.append(averages)
#         attention_map_list.clear()

#     x = np.arange(len(attention_list))
#     plt.figure()
#     plt.plot(x, torch.tensor(attention_list).numpy(), '-o')
#     plt.title("block_level_attention_blink")
#     plt.xlabel("temporal dimension")
#     plt.ylabel("average attention value")
#     plt.grid(True)

#     # Save the plot as an image file
#     plt.savefig("/home/stud/ghuang/Open-Sora/z/" + filename_suffix + "_" + str(token_index) + "_blink.png", dpi=300, bbox_inches="tight")  # Save with high resolution
#     attention_list.clear()



# =====>

# for token_index in range(0, 16):
#     for name in tqdm(sorted_filenames[1624:2000:2]): # 28 depths, each level has one temporal block and one spatial block.
#         attention_map = torch.load(os.path.join(attention_map_path, name), map_location=torch.device('cpu')).float()
#         print(attention_map.shape)

#         # attention_map = attention_map.reshape(1, 16, 2, 30, 45, 80, token_length)
#         attention_map = attention_map[..., :attention_map.shape[2] // 2, :token_length // 2]
#         attention_map = attention_map.reshape(1, 16, 30, 45, 80, token_length // 2)

#         attention_map = attention_map[0]
#         attention_map = attention_map.mean(dim=0) # average across heads
#         attention_map = attention_map.mean(dim=0) # average across frames
#         prefix = f"token_index={token_index}, token={tokens[token_index - 16]},"
#         attention_map = attention_map[..., token_index]

#         attention_map_list.append(attention_map)

#     create_gif_denoising_process(attention_map_list, prefix, filename_suffix + "_" + str(token_index))
#     attention_map_list.clear()
    # break

# create_gif_matplotlib("/home/stud/ghuang/Open-Sora/z/", "/home/stud/ghuang/Open-Sora/visualization_change.gif")
# sys.exit(0)

# for token_index in range(0, token_length // 2):

# =====>
# block_index = 0
# start_number = block_index * 2 # notice that a level have two blocks, so 52 means spatial_block_index=26 

# start_number = 3305
start_number = 1624
end_number = 6000
step = 2
token_index = 2
for i, name in tqdm(enumerate(sorted_filenames[start_number:end_number:step])): # 28 depths, each level has one temporal block and one spatial block.
    attention_map = torch.load(os.path.join(attention_map_path, name), map_location=torch.device('cpu')).float()
    print(attention_map.shape)

    attention_map = attention_map[..., :attention_map.shape[2] // 2, :token_length // 2]
    attention_map = attention_map.reshape(1, 16, 30, 45, 80, token_length // 2)
    attention_map = attention_map[0]
    attention_map = attention_map.mean(dim=0) # average across heads

    # attention_map = attention_map[:attention_map.shape[0] // 2, :token_length // 2]
    # attention_map = attention_map.reshape(30, 45, 80, token_length // 2)

    prefix = f"spatial,denoising_step_no={start_number//56 + 1}/30, block_no={i+1}/28, token_index={token_index}, token={tokens[token_index - token_length // 2]},"
    start_number += step
    attention_map = attention_map[..., token_index]

    # attention_map_list.append(attention_map.mean(dim=0))
    for tensor_index in range(attention_map.shape[0]):
        attention_map_list.append(attention_map[tensor_index])
    create_gif_denoising_process(attention_map_list, prefix, filename_suffix + "_" + str(token_index) + "_" + str(i))

    # attention_map = attention_map.mean(dim=0)
    # attention_map = attention_map[15]
    averages = attention_map.mean(dim=(1, 2))
    x = np.arange(len(averages))
    plt.figure()
    plt.plot(x, averages.numpy(), '-o')
    plt.xticks(ticks=x)
    plt.title(prefix + "attention_blink")
    plt.xlabel("temporal dimension, total={15}")
    plt.ylabel("average attention value")
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig("/home/stud/ghuang/Open-Sora/z/" + filename_suffix + "_" + str(token_index) + "_" + str(i) + "_blink.png", dpi=300, bbox_inches="tight")  # Save with high resolution


    attention_map_list.clear()
    # break

# =====>
# token_index = 0

# path = "/home/stud/ghuang/Open-Sora/z.pt"
# attention_map = torch.load(path, map_location=torch.device('cpu')).float()
# print(attention_map.shape)
# attention_map = attention_map[0]
# attention_map = attention_map.mean(dim=0)

# averages = attention_map.mean(dim=(-2, -1))
# x = np.arange(len(averages))
# plt.figure()
# plt.plot(x, averages.numpy(), '-o')
# plt.title("z_attention_blink")
# plt.xlabel("temporal dimension")
# plt.ylabel("average spatial attention value")
# plt.grid(True)

# plt.savefig("/home/stud/ghuang/Open-Sora/z/z_blink.png", dpi=300, bbox_inches="tight")  # Save with high resolution

# =====>
# prefix = f"spatial,block_no=1/28,token_index=8,token={tokens[token_index - token_length // 2]},"
# create_gif_denoising_process(attention_map_list, prefix, filename_suffix + "_" + str(token_index) + "_" + str(i % 28))

# head_index = 0
# token_index = 2
# for head_index in range(16):
#     for name in tqdm(sorted_filenames[1624:2000:2]): # 28 depths, each level has one temporal block and one spatial block.
#         attention_map = torch.load(os.path.join(attention_map_path, name), map_location=torch.device('cpu')).float()
#         print(attention_map.shape)

#         # attention_map = attention_map.reshape(1, 16, 2, 30, 45, 80, token_length)
#         attention_map = attention_map[..., :attention_map.shape[2] // 2, :token_length // 2]
#         attention_map = attention_map.reshape(1, 16, 30, 45, 80, token_length // 2)

#         attention_map = attention_map[0]
#         attention_map = attention_map[head_index] # average across heads
#         prefix = f"head_index={head_index + 1}, token={tokens[token_index - token_length // 2]},"
#         attention_map = attention_map[..., token_index]

#         for i in range(30):
#             attention_map_list.append(attention_map[i])

#         create_gif_denoising_process(attention_map_list, prefix, filename_suffix + "_" + str(head_index))
#         attention_map_list.clear()
#         break

# =====>
# head_index = 4
# token_index = 0
# current_level = 0
# for name in tqdm(sorted_filenames[1625:2000:2]): # 28 depths, each level has one temporal block and one spatial block.
#     current_level += 1
#     # print(current_level, name)
#     # continue

#     attention_map = torch.load(os.path.join(attention_map_path, name), map_location=torch.device('cpu')).float()
#     print(attention_map.shape)

#     # attention_map = attention_map.reshape(1, 16, 2, 30, 45, 80, token_length)
#     attention_map = attention_map[..., :attention_map.shape[2] // 2, :token_length // 2]
#     attention_map = attention_map.reshape(1, 16, 30, 45, 80, token_length // 2)

#     attention_map = attention_map[0]
#     # attention_map = attention_map[head_index] # average across heads
#     attention_map = attention_map.mean(dim=0)
#     prefix = f"head_index={head_index + 1}, token={tokens[token_index - token_length // 2]}, level={current_level}, "
#     attention_map = attention_map[..., token_index]

#     for i in range(30):
#         attention_map_list.append(attention_map[i])

#     create_gif_denoising_process(attention_map_list, prefix, filename_suffix + "_" + str(current_level))
#     attention_map_list.clear()
#     # break

# =====>
# current_level = 0
# for name in tqdm(sorted_filenames[3304:6000:2]): # 28 depths, each level has one temporal block and one spatial block.
#     current_level += 1

#     attention_map = torch.load(os.path.join(attention_map_path, name), map_location=torch.device('cpu')).float()

#     attention_map = attention_map[:attention_map.shape[0] // 2, :token_length // 2]
#     attention_map = attention_map.reshape(30, 45, 80, token_length // 2)

#     attention_map = attention_map.mean(dim=(1, 2))
#     prefix = f"spatial, head_index=mean, token=mean, last denoising step, "

#     attention_map_list.append(attention_map)

# create_gif_denoising_process(attention_map_list, prefix, filename_suffix + "_" + str(current_level))
# indexes = [22, 38, -1]
# for index in indexes:
#     print(index, tokens[index])