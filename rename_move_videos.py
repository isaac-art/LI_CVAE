import os
from pathlib import Path
import shutil

def process_walk_videos(walks_dir: str, output_dir: str) -> None:
    """
    Rename MP4 files to their parent directory name and move them to output directory.
    
    Args:
        walks_dir: Path to the directory containing numbered walk subdirectories
        output_dir: Path where renamed videos will be moved
    """
    walks_path = Path(walks_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each subdirectory
    for dir_path in walks_path.iterdir():
        if not dir_path.is_dir():
            continue
            
        # Look for MP4 files in the directory
        mp4_files = list(dir_path.glob("*.mp4"))
        if not mp4_files:
            continue
            
        # Get the first MP4 file (assuming there's only one per directory)
        video_file = mp4_files[0]
        
        # Create new filename using directory name
        new_filename = f"{dir_path.name}.mp4"
        target_path = output_path / new_filename
        
        # Move and rename the file
        shutil.move(video_file, target_path)
        print(f"Moved {video_file} to {target_path}")

if __name__ == "__main__":
    walks_directory = "walks"
    output_directory = "walk_videos"
    
    process_walk_videos(walks_directory, output_directory)
    print("Video processing completed!") 