#sample 20000 images randomly from datasets/coco2014/images/train2014 and save it to sampled_images/

import os
import random
import shutil
import tqdm

def sample_coco():
    source_dir = 'datasets/coco2014/images/train2014'
    target_dir = 'sampled_images'
    os.makedirs(target_dir, exist_ok=True)
    files = os.listdir(source_dir)
    random.shuffle(files)
    for i in tqdm.tqdm(range(20000)):
        shutil.copy2(os.path.join(source_dir, files[i]), os.path.join(target_dir, files[i]))

if __name__ == '__main__':
    sample_coco()