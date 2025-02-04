import os
import shutil
import random
import hashlib
import argparse
import yaml
from tqdm import tqdm

def rename_files_in_dir(dir_path):
    """
    Renames image files in a directory using MD5 hash values.
    """
    target_files = [file for file in os.listdir(dir_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

    for file in target_files:
        hash_name = hashlib.md5(file.encode()).hexdigest()
        image_ext = os.path.splitext(file)[1]
        new_name = hash_name + image_ext
        os.rename(os.path.join(dir_path, file), os.path.join(dir_path, new_name))

def process_all_folders(data_dir):
    """
    Renames all image files in the dataset directories.
    """
    folder_list = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    for folder in tqdm(folder_list, desc="Renaming files"):
        rename_files_in_dir(folder)

def create_train_val_split(data_dir, output_dir, train_ratio=0.8):
    """
    Splits images into train and validation sets while preserving class structure.
    The images are copied (not moved) to the output_dir.
    """
    train_path = os.path.join(output_dir, "train")
    val_path = os.path.join(output_dir, "val")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    class_folders = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

    for class_folder in tqdm(class_folders, desc="Splitting data"):
        class_path = os.path.join(data_dir, class_folder)
        train_class_path = os.path.join(train_path, class_folder)
        val_class_path = os.path.join(val_path, class_folder)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        random.shuffle(image_files)
        split_index = int(len(image_files) * train_ratio)
        train_files = image_files[:split_index]
        val_files = image_files[split_index:]

        for file in train_files:
            shutil.copy2(os.path.join(class_path, file), os.path.join(train_class_path, file))

        for file in val_files:
            shutil.copy2(os.path.join(class_path, file), os.path.join(val_class_path, file))
        # 不再删除原文件，保留data_dir中的图片

def create_yolo_config(output_dir):
    """
    Generates a YOLO configuration YAML file.
    """
    train_path = os.path.join(output_dir, "train")
    val_path = os.path.join(output_dir, "val")
    
    class_names = sorted([folder for folder in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, folder))])

    config = {
        "train": train_path,
        "val": val_path,
        "names": class_names
    }

    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Configuration file created at: {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for ViT training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the raw dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output dataset directory.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data (default: 0.8).")
    
    args = parser.parse_args()

    # Step 1: Rename images
    print("\nStep 1: Renaming images...")
    process_all_folders(args.data_dir)

    # Step 2: Split into train and val sets (using copy instead of move)
    print("\nStep 2: Splitting dataset into train and validation sets...")
    create_train_val_split(args.data_dir, args.output_dir, args.train_ratio)

    # Step 3: Create YOLO config file
    print("\nStep 3: Generating YOLO configuration...")
    create_yolo_config(args.output_dir)

    print("\nDataset preprocessing completed!")

if __name__ == "__main__":
    main()
