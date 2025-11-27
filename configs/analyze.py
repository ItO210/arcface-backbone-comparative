import os

# Run this to get the number of classes and images on a dataset.

def count_classes_and_images(dataset_path, image_extensions={'.jpg', '.jpeg', '.png', '.bmp', '.gif'}):
    num_classes = 0
    num_images = 0

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            num_classes += 1
            for file_name in os.listdir(class_path):
                if os.path.splitext(file_name)[1].lower() in image_extensions:
                    num_images += 1

    return num_classes, num_images

dataset_path = "path/to/dataset_folder"
classes, images = count_classes_and_images(dataset_path)
print(f"Number of classes: {classes}")
print(f"Number of images: {images}")
