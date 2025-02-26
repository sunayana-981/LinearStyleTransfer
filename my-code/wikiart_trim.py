import os
import random
import shutil
from tqdm import tqdm

# Define the source and destination directories
source_dir = "datasets/wikiArt/wikiart/Ukiyo_e"
destination_dir = 'sampled_images/Ukiyo-e/'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get a list of all image files in the source directory
image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Randomly sample 40,000 images
sampled_images = random.sample(image_files, 100)

# Copy the sampled images to the destination directory with a progress bar
for image in tqdm(sampled_images, desc="Copying images"):
    shutil.copy(os.path.join(source_dir, image), os.path.join(destination_dir, image))

print(f'Successfully sampled 100 images to {destination_dir}')