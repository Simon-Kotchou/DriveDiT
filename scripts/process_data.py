#!/usr/bin/env python3
"""
Data preprocessing script for large-scale video datasets.
Converts videos to memory-mapped format for efficient training.
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.pipeline import VideoMemoryMap


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DriveDiT Data Processing')
    
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing videos')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory (default: output_dir/cache)')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(),
                       help='Number of parallel workers')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.mp4', '.avi', '.mov', '.mkv', '.webm'],
                       help='Video file extensions to process')
    parser.add_argument('--recursive', action='store_true',
                       help='Search for videos recursively')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing cache files')
    
    return parser.parse_args()


def find_video_files(input_dir: str, extensions: List[str], recursive: bool = False) -> List[Path]:
    """Find all video files in input directory."""
    input_path = Path(input_dir)
    video_files = []
    
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        if recursive:
            video_files.extend(input_path.rglob(f"*{ext}"))
        else:
            video_files.extend(input_path.glob(f"*{ext}"))
    
    return sorted(video_files)


def process_single_video(args_tuple: Tuple[str, str, bool]) -> Tuple[str, bool, str]:
    """Process a single video file."""
    video_path, cache_dir, overwrite = args_tuple
    
    try:
        # Check if cache already exists
        video_name = Path(video_path).stem
        cache_path = Path(cache_dir) / f"{video_name}.idx"
        
        if cache_path.exists() and not overwrite:
            return video_path, True, "Already cached"
        
        # Create memory-mapped cache
        start_time = time.time()
        video_mmap = VideoMemoryMap(video_path, cache_dir)
        process_time = time.time() - start_time
        
        return video_path, True, f"Processed in {process_time:.2f}s ({len(video_mmap)} frames)"
    
    except Exception as e:
        return video_path, False, str(e)


def print_progress(completed: int, total: int, successful: int, failed: int):
    """Print processing progress."""
    progress = completed / total * 100
    print(f"\\rProgress: {completed}/{total} ({progress:.1f}%) | "
          f"Success: {successful} | Failed: {failed}", end="", flush=True)


def main():
    """Main data processing function."""
    args = parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("DriveDiT Data Processing")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Extensions: {args.extensions}")
    print(f"Recursive: {args.recursive}")
    print(f"Overwrite: {args.overwrite}")
    print("=" * 50)
    
    # Find video files
    print("Searching for video files...")
    video_files = find_video_files(input_dir, args.extensions, args.recursive)
    
    if not video_files:
        print("No video files found!")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Prepare arguments for parallel processing
    process_args = [(str(video_path), str(cache_dir), args.overwrite) 
                   for video_path in video_files]
    
    # Process videos in parallel
    print(f"Processing videos with {args.num_workers} workers...")
    start_time = time.time()
    
    successful = 0
    failed = 0
    failed_files = []
    
    with mp.Pool(args.num_workers) as pool:
        for i, (video_path, success, message) in enumerate(pool.imap(process_single_video, process_args)):
            if success:
                successful += 1
            else:
                failed += 1
                failed_files.append((video_path, message))
            
            # Print progress every 10 files or at the end
            if (i + 1) % 10 == 0 or (i + 1) == len(video_files):
                print_progress(i + 1, len(video_files), successful, failed)
    
    total_time = time.time() - start_time
    print(f"\\n\\nProcessing completed in {total_time:.2f} seconds")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed_files:
        print("\\nFailed files:")
        for video_path, error in failed_files:
            print(f"  {video_path}: {error}")
    
    # Calculate statistics
    if successful > 0:
        cache_files = list(cache_dir.glob("*.idx"))
        total_cache_size = sum(f.stat().st_size for f in cache_files) / (1024**3)  # GB
        avg_time_per_video = total_time / successful
        
        print(f"\\nStatistics:")
        print(f"  Cache size: {total_cache_size:.2f} GB")
        print(f"  Average time per video: {avg_time_per_video:.2f} seconds")
        print(f"  Processing rate: {successful / total_time:.2f} videos/second")
    
    # Create dataset summary
    summary_path = output_dir / "dataset_summary.json"
    summary = {
        "input_directory": str(input_dir),
        "cache_directory": str(cache_dir),
        "total_videos": len(video_files),
        "successful": successful,
        "failed": failed,
        "processing_time": total_time,
        "cache_size_gb": total_cache_size if successful > 0 else 0,
        "failed_files": [{"path": path, "error": error} for path, error in failed_files]
    }
    
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nDataset summary saved to: {summary_path}")


if __name__ == "__main__":
    main()