import os
import shutil
from pathlib import Path

def flatten_image_directories(source_dir: str, target_dir: str) -> None:
    """
    Flatten nested image directories by moving only the deepest directories containing PNG files.
    
    Args:
        source_dir: Path to the source directory containing nested directories
        target_dir: Path to the target directory where flattened structure will be created
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Keep track of directories that contain PNGs directly
    dirs_with_pngs = []
    
    # First pass: find all directories that contain PNG files
    for root, _, files in os.walk(source_path):
        if any(file.lower().endswith('.png') for file in files):
            dirs_with_pngs.append(Path(root))
    
    # Second pass: only keep the deepest directories
    deepest_dirs = []
    for dir_path in dirs_with_pngs:
        # Check if this directory is a parent of any other directory in the list
        if not any(other_dir.is_relative_to(dir_path) for other_dir in dirs_with_pngs if other_dir != dir_path):
            deepest_dirs.append(dir_path)
    
    # Move files from deepest directories
    for dir_path in deepest_dirs:
        # Create new directory name by joining path components with underscores
        rel_path = dir_path.relative_to(source_path)
        new_dir_name = '_'.join(rel_path.parts)
        
        # Create new directory in target
        new_dir_path = target_path / new_dir_name
        new_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Move all PNG files to the new directory
        for file in os.listdir(dir_path):
            if file.lower().endswith('.png'):
                source_file = dir_path / file
                target_file = new_dir_path / file
                shutil.copy2(source_file, target_file)

if __name__ == "__main__":
    source_directory = "li_art_catagory"
    target_directory = "inputs"
    
    flatten_image_directories(source_directory, target_directory)
    print("Directory structure flattened successfully!") 