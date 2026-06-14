import os, random, shutil

def trim_class(folder, keep):
    images = os.listdir(folder)
    if len(images) <= keep:
        print(f"{folder} already under {keep}, skipping")
        return
    to_delete = random.sample(images, len(images) - keep)
    for f in to_delete:
        os.remove(os.path.join(folder, f))
    print(f"Trimmed {folder} to {keep} images")

trim_class("dataset_all/Car/n04285008",   900)