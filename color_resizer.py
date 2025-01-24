import os
from PIL import Image

def resize_and_crop(image_path, output_path, size=(128, 128)):
    with Image.open(image_path) as img:
        # Calculate the resize dimension
        img_ratio = img.width / img.height
        target_ratio = size[0] / size[1]

        if img_ratio > target_ratio:
            # Image is wider than target aspect ratio
            new_height = size[1]
            new_width = int(new_height * img_ratio)
        else:
            # Image is taller than target aspect ratio
            new_width = size[0]
            new_height = int(new_width / img_ratio)

        # Resize the image
        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Calculate crop position
        left = (new_width - size[0]) / 2
        top = (new_height - size[1]) / 2
        right = (new_width + size[0]) / 2
        bottom = (new_height + size[1]) / 2

        # Crop the image
        img = img.crop((left, top, right, bottom))

        # Save the image
        img.save(output_path)

def process_images(input_folder, output_folder, size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_and_crop(input_path, output_path, size)

# Example usage
input_folder = 'color/swatches_320/'
output_folder = 'color/swatches_128/'
process_images(input_folder, output_folder)
