"""
Intelligent video chunking for hours-long driving videos.
Implements temporal overlap, scene detection, and efficient sampling strategies.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator
import json
import hashlib
from dataclasses import dataclass
from collections import defaultdict
import time

from .pipeline import VideoMemoryMap


@dataclass
class ChunkConfig:
    """Configuration for video chunking."""
    chunk_duration: int = 300  # 5 minutes in seconds
    overlap_duration: int = 30  # 30 seconds overlap
    min_chunk_duration: int = 60  # Minimum 1 minute chunks
    target_fps: float = 10.0  # Target FPS for training
    
    # Scene detection
    scene_threshold: float = 0.3  # Scene change threshold
    use_scene_detection: bool = True
    
    # Quality filtering
    min_motion_threshold: float = 0.01  # Filter static scenes
    blur_threshold: float = 100.0  # Filter blurry frames
    
    # Temporal sampling strategies
    sampling_strategy: str = "uniform"  # uniform, adaptive, scene_aware
    adaptive_motion_weight: float = 0.3  # Weight for motion-based sampling


class VideoChunker:
    """Intelligent video chunking with scene detection and quality filtering."""
    
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.scene_detector = SceneDetector(config.scene_threshold)
        self.quality_filter = QualityFilter(
            config.min_motion_threshold,
            config.blur_threshold
        )
    
    def chunk_video(
        self, 
        video_path: str, 
        output_dir: str,
        cache_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Chunk a long video into training segments.
        
        Returns:
            List of chunk metadata dictionaries
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load video
        print(f"Processing video: {video_path}")
        video_mmap = VideoMemoryMap(str(video_path), cache_dir)
        
        # Calculate temporal parameters
        total_frames = len(video_mmap)
        fps = video_mmap.fps
        total_duration = total_frames / fps
        
        print(f"Video stats: {total_frames} frames, {fps:.2f} FPS, {total_duration:.1f}s")
        
        # Detect scenes if enabled
        scene_boundaries = []
        if self.config.use_scene_detection:
            print("Detecting scene boundaries...")
            scene_boundaries = self.scene_detector.detect_scenes(video_mmap)
            print(f"Found {len(scene_boundaries)} scene boundaries")
        
        # Generate chunk boundaries
        chunk_boundaries = self._generate_chunk_boundaries(
            total_duration, scene_boundaries
        )
        
        # Create chunks
        chunks = []
        for i, (start_time, end_time) in enumerate(chunk_boundaries):
            chunk_info = self._create_chunk(
                video_mmap, 
                start_time, 
                end_time, 
                output_dir, 
                f"{video_path.stem}_chunk_{i:04d}"
            )
            if chunk_info:
                chunks.append(chunk_info)
        
        # Save chunk index
        chunk_index = {
            "source_video": str(video_path),
            "total_duration": total_duration,
            "total_chunks": len(chunks),
            "config": self.config.__dict__,
            "chunks": chunks
        }
        
        index_path = output_dir / f"{video_path.stem}_chunks.json"
        with open(index_path, 'w') as f:
            json.dump(chunk_index, f, indent=2)
        
        print(f"Created {len(chunks)} chunks, index saved to {index_path}")
        return chunks
    
    def _generate_chunk_boundaries(
        self, 
        total_duration: float, 
        scene_boundaries: List[float]
    ) -> List[Tuple[float, float]]:
        """Generate chunk start/end times with scene awareness."""
        boundaries = []
        
        if not self.config.use_scene_detection or not scene_boundaries:
            # Simple uniform chunking
            start = 0.0
            while start < total_duration:
                end = min(start + self.config.chunk_duration, total_duration)
                if end - start >= self.config.min_chunk_duration:
                    boundaries.append((start, end))
                start += self.config.chunk_duration - self.config.overlap_duration
        else:
            # Scene-aware chunking
            boundaries = self._scene_aware_chunking(total_duration, scene_boundaries)
        
        return boundaries
    
    def _scene_aware_chunking(
        self, 
        total_duration: float, 
        scene_boundaries: List[float]
    ) -> List[Tuple[float, float]]:
        """Create chunks that respect scene boundaries when possible."""
        boundaries = []
        current_start = 0.0
        
        for scene_end in scene_boundaries + [total_duration]:
            # Check if we can extend current chunk to this scene boundary
            if scene_end - current_start <= self.config.chunk_duration * 1.5:
                # Extend to scene boundary
                if scene_end - current_start >= self.config.min_chunk_duration:
                    boundaries.append((current_start, scene_end))
                    current_start = max(0, scene_end - self.config.overlap_duration)
            else:
                # Scene is too long, break it into regular chunks
                chunk_start = current_start
                while chunk_start < scene_end:
                    chunk_end = min(chunk_start + self.config.chunk_duration, scene_end)
                    if chunk_end - chunk_start >= self.config.min_chunk_duration:
                        boundaries.append((chunk_start, chunk_end))
                    chunk_start = chunk_end - self.config.overlap_duration
                
                current_start = max(0, scene_end - self.config.overlap_duration)
        
        return boundaries
    
    def _create_chunk(
        self,
        video_mmap: VideoMemoryMap,
        start_time: float,
        end_time: float,
        output_dir: Path,
        chunk_name: str
    ) -> Optional[Dict]:
        """Create a single chunk with quality filtering."""
        fps = video_mmap.fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Sample frames for quality assessment
        sample_indices = np.linspace(start_frame, end_frame-1, min(10, end_frame-start_frame), dtype=int)
        sample_frames = [video_mmap.get_frame(idx) for idx in sample_indices]
        
        # Quality filtering
        quality_score = self.quality_filter.assess_quality(sample_frames)
        if quality_score < 0.5:  # Skip low-quality chunks
            print(f"Skipping low-quality chunk {chunk_name} (score: {quality_score:.3f})")
            return None
        
        # Temporal sampling based on strategy
        if self.config.sampling_strategy == "uniform":
            frame_indices = self._uniform_sampling(start_frame, end_frame)
        elif self.config.sampling_strategy == "adaptive":
            frame_indices = self._adaptive_sampling(video_mmap, start_frame, end_frame)
        else:  # scene_aware
            frame_indices = self._scene_aware_sampling(video_mmap, start_frame, end_frame)
        
        # Create chunk file
        chunk_path = output_dir / f"{chunk_name}.npz"
        frames = np.stack([video_mmap.get_frame(idx) for idx in frame_indices])
        
        # Save chunk
        np.savez_compressed(
            chunk_path,
            frames=frames,
            frame_indices=frame_indices,
            timestamps=np.array(frame_indices) / fps,
            quality_score=quality_score
        )
        
        return {
            "name": chunk_name,
            "path": str(chunk_path),
            "start_time": start_time,
            "end_time": end_time,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_frames": len(frame_indices),
            "duration": end_time - start_time,
            "quality_score": quality_score,
            "frame_indices": frame_indices.tolist()
        }
    
    def _uniform_sampling(self, start_frame: int, end_frame: int) -> np.ndarray:
        """Uniform temporal sampling."""
        total_frames = end_frame - start_frame
        target_frames = int((total_frames / self.config.target_fps) * self.config.target_fps)
        target_frames = min(target_frames, total_frames)
        
        return np.linspace(start_frame, end_frame-1, target_frames, dtype=int)
    
    def _adaptive_sampling(self, video_mmap: VideoMemoryMap, start_frame: int, end_frame: int) -> np.ndarray:
        """Adaptive sampling based on motion content."""
        # Sample every N frames to analyze motion
        analysis_step = max(1, (end_frame - start_frame) // 100)
        analysis_frames = range(start_frame, end_frame, analysis_step)
        
        # Calculate motion scores
        motion_scores = []
        prev_frame = None
        
        for frame_idx in analysis_frames:
            frame = video_mmap.get_frame(frame_idx)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                motion_score = np.mean(diff) / 255.0
                motion_scores.append(motion_score)
            
            prev_frame = gray
        
        if not motion_scores:
            return self._uniform_sampling(start_frame, end_frame)
        
        # Create sampling probability based on motion
        motion_scores = np.array(motion_scores)
        motion_weights = motion_scores / (np.mean(motion_scores) + 1e-8)
        motion_weights = np.clip(motion_weights, 0.1, 3.0)  # Limit extreme weights
        
        # Weighted sampling
        total_frames = end_frame - start_frame
        target_frames = int(total_frames * self.config.target_fps / video_mmap.fps)
        
        # Create cumulative weights for sampling
        frame_weights = np.interp(
            np.arange(start_frame, end_frame),
            analysis_frames[1:],  # Skip first frame (no motion score)
            motion_weights
        )
        frame_weights[0] = motion_weights[0] if motion_weights.size > 0 else 1.0
        
        # Sample frames based on weights
        cumulative_weights = np.cumsum(frame_weights)
        cumulative_weights = cumulative_weights / cumulative_weights[-1]
        
        sample_points = np.linspace(0, 1, target_frames)
        sampled_indices = []
        
        for point in sample_points:
            idx = np.searchsorted(cumulative_weights, point)
            sampled_indices.append(start_frame + min(idx, len(frame_weights) - 1))
        
        return np.array(sampled_indices)
    
    def _scene_aware_sampling(self, video_mmap: VideoMemoryMap, start_frame: int, end_frame: int) -> np.ndarray:
        """Scene-aware sampling that preserves important transitions."""
        # For now, fall back to uniform sampling
        # In full implementation, detect scene changes within chunk
        return self._uniform_sampling(start_frame, end_frame)


class SceneDetector:
    """Detect scene boundaries in video using histogram comparison."""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    def detect_scenes(self, video_mmap: VideoMemoryMap) -> List[float]:
        """Detect scene boundaries in video."""
        boundaries = []
        total_frames = len(video_mmap)
        step = max(1, total_frames // 1000)  # Sample every N frames
        
        prev_hist = None
        
        for i in range(0, total_frames, step):
            frame = video_mmap.get_frame(i)
            hist = self._compute_histogram(frame)
            
            if prev_hist is not None:
                similarity = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
                if similarity < (1.0 - self.threshold):
                    boundaries.append(i / video_mmap.fps)
            
            prev_hist = hist
        
        return boundaries
    
    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for frame."""
        # Convert to HSV for better scene detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()


class QualityFilter:
    """Filter video chunks based on quality metrics."""
    
    def __init__(self, min_motion_threshold: float = 0.01, blur_threshold: float = 100.0):
        self.min_motion_threshold = min_motion_threshold
        self.blur_threshold = blur_threshold
    
    def assess_quality(self, frames: List[np.ndarray]) -> float:
        """Assess quality of a sequence of frames."""
        if len(frames) < 2:
            return 0.0
        
        motion_score = self._assess_motion(frames)
        sharpness_score = self._assess_sharpness(frames)
        brightness_score = self._assess_brightness(frames)
        
        # Weighted combination
        quality_score = (
            0.4 * motion_score +
            0.4 * sharpness_score +
            0.2 * brightness_score
        )
        
        return quality_score
    
    def _assess_motion(self, frames: List[np.ndarray]) -> float:
        """Assess motion content in frames."""
        motion_scores = []
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            diff = cv2.absdiff(prev_gray, curr_gray)
            motion = np.mean(diff) / 255.0
            motion_scores.append(motion)
        
        avg_motion = np.mean(motion_scores)
        
        # Score based on motion level
        if avg_motion < self.min_motion_threshold:
            return 0.0  # Too static
        elif avg_motion > 0.1:
            return 1.0  # Good motion
        else:
            return avg_motion / 0.1  # Scaled score
    
    def _assess_sharpness(self, frames: List[np.ndarray]) -> float:
        """Assess sharpness/blur in frames."""
        sharpness_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharpness_scores.append(sharpness)
        
        avg_sharpness = np.mean(sharpness_scores)
        
        # Score based on sharpness
        if avg_sharpness < self.blur_threshold:
            return 0.0  # Too blurry
        else:
            return min(1.0, avg_sharpness / (self.blur_threshold * 5))
    
    def _assess_brightness(self, frames: List[np.ndarray]) -> float:
        """Assess brightness levels in frames."""
        brightness_scores = []
        
        for frame in frames:
            brightness = np.mean(frame) / 255.0
            # Prefer brightness in [0.2, 0.8] range
            if 0.2 <= brightness <= 0.8:
                score = 1.0
            elif brightness < 0.1 or brightness > 0.9:
                score = 0.0  # Too dark or too bright
            else:
                score = 0.5  # Acceptable
            
            brightness_scores.append(score)
        
        return np.mean(brightness_scores)


class ChunkedVideoDataset:
    """Dataset that efficiently handles chunked videos."""
    
    def __init__(self, chunk_dir: str, sequence_length: int = 16):
        self.chunk_dir = Path(chunk_dir)
        self.sequence_length = sequence_length
        
        # Load chunk indices
        self.chunks = []
        for index_file in self.chunk_dir.glob("*_chunks.json"):
            with open(index_file, 'r') as f:
                chunk_data = json.load(f)
                self.chunks.extend(chunk_data["chunks"])
        
        # Filter chunks by minimum length
        self.chunks = [c for c in self.chunks if c["num_frames"] >= sequence_length]
        
        print(f"Loaded {len(self.chunks)} valid chunks")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        # Load chunk data
        chunk_data = np.load(chunk["path"])
        frames = chunk_data["frames"]
        
        # Sample sequence
        if len(frames) > self.sequence_length:
            start_idx = np.random.randint(0, len(frames) - self.sequence_length + 1)
            sequence = frames[start_idx:start_idx + self.sequence_length]
        else:
            sequence = frames
        
        # Convert to tensor and normalize
        sequence = torch.from_numpy(sequence).float() / 255.0
        sequence = sequence.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        
        return {
            "frames": sequence,
            "chunk_info": chunk
        }


def process_long_videos(
    input_dir: str,
    output_dir: str,
    config: Optional[ChunkConfig] = None
) -> None:
    """Process all long videos in a directory."""
    if config is None:
        config = ChunkConfig()
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(video_files)} videos to process")
    
    chunker = VideoChunker(config)
    total_chunks = 0
    
    for video_file in video_files:
        try:
            print(f"\nProcessing: {video_file}")
            chunks = chunker.chunk_video(
                str(video_file),
                str(output_path),
                cache_dir=str(output_path / "cache")
            )
            total_chunks += len(chunks)
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
    
    print(f"\nProcessing complete. Created {total_chunks} total chunks.")


if __name__ == "__main__":
    # Example usage
    config = ChunkConfig(
        chunk_duration=300,  # 5 minutes
        overlap_duration=30,  # 30 seconds
        use_scene_detection=True,
        sampling_strategy="adaptive"
    )
    
    process_long_videos(
        input_dir="./data/raw_videos",
        output_dir="./data/chunked_videos",
        config=config
    )