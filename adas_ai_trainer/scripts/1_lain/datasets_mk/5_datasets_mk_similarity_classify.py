import os
import random
import shutil

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    name_list = []
    size_list = []
    total_size = 0  # in bytes
    size_limit = 500 * 1024 * 1024  # 500 MB

    folders = sorted(os.listdir(input_folder))
    loop_count = 0

    while total_size <= size_limit:
        loop_count += 1
        all_folders_processed = True

        for folder in folders:
            folder_path = os.path.join(input_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

            unprocessed_images = [img for img in images if os.path.splitext(img)[0] not in name_list]

            if not unprocessed_images:
                continue

            all_folders_processed = False

            # Select a random image
            image = random.choice(unprocessed_images)
            image_path = os.path.join(folder_path, image)

            # Extract name without extension
            name = os.path.splitext(image)[0]
            name_list.append(name)

            # Get image size
            size = os.path.getsize(image_path)
            size_list.append(size)
            total_size += size

            print(f"Selected: {image} | Current total size: {total_size / (1024 * 1024):.2f} MB")

            if total_size > size_limit:
                print("Size limit exceeded. Stopping file selection.")
                break

        if all_folders_processed:
            print("All folders processed. Restarting from the first folder.")
            continue

        if total_size > size_limit:
            break

    # Copy selected files to the output folder
    current_size = 0

    for folder in folders:
        folder_path = os.path.join(input_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image in images:
            image_path = os.path.join(folder_path, image)
            name = os.path.splitext(image)[0]

            if name not in name_list:
                continue

            target_path = os.path.join(output_folder, image)
            shutil.copy(image_path, target_path)
            current_size += os.path.getsize(image_path)

            print(f"Copied: {image} | From folder: {folder} | Current output size: {current_size / (1024 * 1024):.2f} MB")

            if current_size > size_limit:
                print("Output folder size limit reached.")
                print(f"Total folders: {len(folders)}")
                print(f"Number of loops: {loop_count}")
                return

    print(f"Total folders: {len(folders)}")
    print(f"Number of loops: {loop_count}")

if __name__ == "__main__":
    input_folder = "/data_KTP/승용_자율주행차_데이터셋/output_groups/"
    output_folder = "/data_KTP/승용_자율주행차_데이터셋/output_groups_select"

    process_images(input_folder, output_folder)
