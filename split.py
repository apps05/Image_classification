import os, shutil, random

SOURCE_DIR = "dataset_all"   # current folder with all images
OUTPUT_DIR = "dataset"
VAL_SPLIT  = 0.2             # 20% for validation

classes = ["Bus", "Truck", "Car", "Bike", "None"]

for cls in classes:
    src = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(src)
    random.shuffle(images)

    val_count  = int(len(images) * VAL_SPLIT)
    val_imgs   = images[:val_count]
    train_imgs = images[val_count:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        dst = os.path.join(OUTPUT_DIR, split, cls)
        os.makedirs(dst, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    print(f"{cls}: {len(train_imgs)} train, {len(val_imgs)} val")