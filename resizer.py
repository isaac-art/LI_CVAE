import os
from pathlib import Path
from PIL import Image
import numpy as np

def resize_and_pad_image(image: Image.Image, target_size: int = 128) -> Image.Image:
    """Resize image keeping aspect ratio and pad to square."""
    # Get original dimensions
    width, height = image.size
    
    # Calculate new dimensions preserving aspect ratio
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
        
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to binary (black and white)
    resized_image = resized_image.point(lambda x: 0 if x < 190 else 255, '1')
    
    # Create new square white image
    padded_image = Image.new('1', (target_size, target_size), 1)
    
    # Calculate padding
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    
    # Paste resized image onto padded background
    padded_image.paste(resized_image, (left, top))
    
    return padded_image

def process_images(image_dir: str = "inputs") -> None:
    """Process all PNG images in directory and subdirectories."""
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{image_dir}' not found")
        
    # Process all PNG files in directory and subdirectories
    for img_path in image_dir.rglob("*.png"):
        try:
            # Open and convert to grayscale
            with Image.open(img_path) as img:
                img = img.convert('L')
                
                # Resize and pad
                processed_img = resize_and_pad_image(img)
                
                # Save back to same location
                processed_img.save(img_path)
                print(f"Processed: {img_path}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    process_images()
