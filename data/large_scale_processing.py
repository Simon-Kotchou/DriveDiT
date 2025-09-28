"""
Large-scale dataset processing strategy for 100k+ hours of driving data.
Implements hierarchical storage, distributed processing, and efficient data management.
"""

import os
import shutil
import hashlib
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading
from queue import Queue
import logging
import psutil
import tempfile

import torch
import numpy as np
import cv2
from tqdm import tqdm

from .video_chunking import VideoChunker, ChunkConfig, QualityFilter
from .pipeline import VideoMemoryMap


@dataclass
class StorageTier:
    """Configuration for storage tier."""
    name: str
    path: str
    capacity_gb: float
    access_speed: str  # "fast", "medium", "slow"
    cost_per_gb: float
    retention_days: int


@dataclass
class DatasetMetadata:
    """Metadata for dataset organization."""
    dataset_id: str
    source: str  # "waymo", "nuscenes", "bdd100k", "custom"
    total_hours: float
    total_size_gb: float
    fps: float
    resolution: Tuple[int, int]
    num_sequences: int
    quality_score: float
    processing_date: str
    storage_tier: str


@dataclass
class ProcessingConfig:
    """Configuration for large-scale processing."""
    # Processing parallelism
    num_workers: int = mp.cpu_count()
    max_concurrent_videos: int = 100
    chunk_size_gb: float = 10.0  # Process in 10GB chunks
    
    # Storage tiers
    storage_tiers: List[StorageTier] = None
    
    # Quality filtering
    min_quality_score: float = 0.7
    max_blur_ratio: float = 0.3
    min_motion_score: float = 0.01
    
    # Hierarchical organization
    organize_by_quality: bool = True
    organize_by_scenario: bool = True
    organize_by_time_of_day: bool = True
    
    # Processing pipeline
    enable_deduplication: bool = True
    enable_compression: bool = True
    compression_quality: int = 85
    
    # Database tracking
    use_database: bool = True
    database_path: str = "./dataset_metadata.db"


class HierarchicalStorageManager:
    """Manage data across multiple storage tiers based on access patterns."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tiers = config.storage_tiers or self._create_default_tiers()
        self.access_tracker = {}  # Track file access patterns
        
        # Create tier directories
        for tier in self.tiers:
            Path(tier.path).mkdir(parents=True, exist_ok=True)
    
    def _create_default_tiers(self) -> List[StorageTier]:
        """Create default storage tier configuration."""
        return [
            StorageTier(
                name="nvme_hot",
                path="./data/tier1_nvme",
                capacity_gb=1000,
                access_speed="fast",
                cost_per_gb=1.0,
                retention_days=30
            ),
            StorageTier(
                name="ssd_warm",
                path="./data/tier2_ssd", 
                capacity_gb=10000,
                access_speed="medium",
                cost_per_gb=0.3,
                retention_days=180
            ),
            StorageTier(
                name="hdd_cold",
                path="./data/tier3_hdd",
                capacity_gb=100000,
                access_speed="slow", 
                cost_per_gb=0.05,
                retention_days=365 * 5
            )
        ]
    
    def allocate_storage(self, file_size_gb: float, priority: str = "medium") -> StorageTier:
        """Allocate storage tier based on size and priority."""
        
        # Priority mapping to tier preference
        tier_preferences = {
            "high": ["nvme_hot", "ssd_warm", "hdd_cold"],
            "medium": ["ssd_warm", "hdd_cold", "nvme_hot"], 
            "low": ["hdd_cold", "ssd_warm", "nvme_hot"]
        }
        
        preferences = tier_preferences.get(priority, tier_preferences["medium"])
        
        for tier_name in preferences:
            tier = next(t for t in self.tiers if t.name == tier_name)
            if self._check_tier_capacity(tier, file_size_gb):
                return tier
        
        # Fallback to largest capacity tier
        return max(self.tiers, key=lambda t: t.capacity_gb)
    
    def _check_tier_capacity(self, tier: StorageTier, required_gb: float) -> bool:
        """Check if tier has sufficient capacity."""
        try:
            tier_path = Path(tier.path)
            if tier_path.exists():
                usage = shutil.disk_usage(tier_path)
                available_gb = usage.free / (1024**3)
                return available_gb > required_gb * 1.1  # 10% buffer
            return True  # Assume available if path doesn't exist
        except:
            return False
    
    def promote_file(self, file_path: str, target_tier: str):
        """Promote frequently accessed file to faster tier."""
        source_path = Path(file_path)
        target_tier_obj = next(t for t in self.tiers if t.name == target_tier)
        target_path = Path(target_tier_obj.path) / source_path.name
        
        if source_path != target_path:
            shutil.move(str(source_path), str(target_path))
            self._update_access_tracker(str(target_path))
    
    def _update_access_tracker(self, file_path: str):
        """Update file access tracking."""
        if file_path not in self.access_tracker:
            self.access_tracker[file_path] = {
                'access_count': 0,
                'last_access': time.time(),
                'creation_time': time.time()
            }
        
        self.access_tracker[file_path]['access_count'] += 1
        self.access_tracker[file_path]['last_access'] = time.time()


class DatasetDatabase:
    """SQLite database for tracking dataset metadata and processing status."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    total_hours REAL,
                    total_size_gb REAL,
                    fps REAL,
                    resolution TEXT,
                    num_sequences INTEGER,
                    quality_score REAL,
                    processing_date TEXT,
                    storage_tier TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_files (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    file_path TEXT,
                    file_size_gb REAL,
                    duration_seconds REAL,
                    quality_score REAL,
                    processing_status TEXT,
                    chunk_ids TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT,
                    status TEXT,
                    input_files TEXT,
                    output_files TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_source ON datasets(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_dataset ON video_files(dataset_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_status ON video_files(processing_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON processing_jobs(status)")
    
    def insert_dataset(self, metadata: DatasetMetadata):
        """Insert dataset metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.dataset_id, metadata.source, metadata.total_hours,
                metadata.total_size_gb, metadata.fps, f"{metadata.resolution[0]}x{metadata.resolution[1]}",
                metadata.num_sequences, metadata.quality_score, metadata.processing_date,
                metadata.storage_tier, json.dumps(asdict(metadata))
            ))
    
    def insert_video_file(self, video_id: str, dataset_id: str, file_path: str, 
                         file_size_gb: float, duration: float, quality_score: float):
        """Insert video file record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO video_files 
                (id, dataset_id, file_path, file_size_gb, duration_seconds, quality_score, processing_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (video_id, dataset_id, file_path, file_size_gb, duration, quality_score, "pending"))
    
    def update_processing_status(self, video_id: str, status: str, chunk_ids: List[str] = None):
        """Update video processing status."""
        with sqlite3.connect(self.db_path) as conn:
            chunk_ids_str = json.dumps(chunk_ids) if chunk_ids else None
            conn.execute("""
                UPDATE video_files 
                SET processing_status = ?, chunk_ids = ?
                WHERE id = ?
            """, (status, chunk_ids_str, video_id))
    
    def get_pending_videos(self, limit: int = 100) -> List[Tuple]:
        """Get videos pending processing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, file_path, file_size_gb, duration_seconds
                FROM video_files 
                WHERE processing_status = 'pending'
                LIMIT ?
            """, (limit,))
            return cursor.fetchall()
    
    def get_dataset_stats(self) -> Dict:
        """Get overall dataset statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_datasets,
                    SUM(total_hours) as total_hours,
                    SUM(total_size_gb) as total_size_gb,
                    AVG(quality_score) as avg_quality
                FROM datasets
            """)
            stats = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            # Video file stats
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_videos,
                    SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as completed_videos,
                    SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) as failed_videos
                FROM video_files
            """)
            video_stats = dict(zip([col[0] for col in cursor.description], cursor.fetchone()))
            
            return {**stats, **video_stats}


class LargeScaleProcessor:
    """Main processor for 100k+ hour datasets."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.storage_manager = HierarchicalStorageManager(config)
        self.database = DatasetDatabase(config.database_path) if config.use_database else None
        self.quality_filter = QualityFilter()
        
        # Processing state
        self.processed_count = 0
        self.failed_count = 0
        self.total_size_processed = 0.0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_dataset_directory(self, input_dir: str, dataset_source: str = "custom") -> str:
        """Process entire dataset directory with hierarchical organization."""
        input_path = Path(input_dir)
        
        self.logger.info(f"Starting large-scale processing of {input_dir}")
        
        # Discover all video files
        video_files = self._discover_video_files(input_path)
        self.logger.info(f"Discovered {len(video_files)} video files")
        
        # Estimate total processing requirements
        total_size_gb = sum(f.stat().st_size for f in video_files) / (1024**3)
        estimated_hours = self._estimate_total_hours(video_files)
        
        self.logger.info(f"Estimated dataset: {estimated_hours:.1f} hours, {total_size_gb:.1f} GB")
        
        # Create dataset metadata
        dataset_id = self._generate_dataset_id(dataset_source, input_path.name)
        
        # Organize processing into batches
        processing_batches = self._create_processing_batches(video_files)
        
        # Process batches
        output_paths = []
        for batch_idx, batch in enumerate(processing_batches):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(processing_batches)}")
            
            batch_output = self._process_batch(
                batch, 
                dataset_id, 
                dataset_source,
                batch_idx
            )
            output_paths.extend(batch_output)
            
            # Memory cleanup between batches
            self._cleanup_memory()
        
        # Create final dataset metadata
        self._finalize_dataset_metadata(dataset_id, dataset_source, output_paths)
        
        # Generate processing report
        report_path = self._generate_processing_report(dataset_id)
        
        self.logger.info(f"Processing completed. Report: {report_path}")
        return report_path
    
    def _discover_video_files(self, input_path: Path) -> List[Path]:
        """Recursively discover all video files."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(input_path.rglob(f'*{ext}'))
        
        # Filter by size (skip very small files)
        video_files = [f for f in video_files if f.stat().st_size > 100 * 1024 * 1024]  # > 100MB
        
        return sorted(video_files)
    
    def _estimate_total_hours(self, video_files: List[Path]) -> float:
        """Estimate total hours by sampling video durations."""
        if not video_files:
            return 0.0
        
        # Sample subset for duration estimation
        sample_size = min(100, len(video_files))
        sample_files = np.random.choice(video_files, sample_size, replace=False)
        
        total_duration = 0.0
        valid_samples = 0
        
        for video_file in sample_files:
            try:
                cap = cv2.VideoCapture(str(video_file))
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps > 0:
                    duration = frame_count / fps
                    total_duration += duration
                    valid_samples += 1
                cap.release()
            except:
                continue
        
        if valid_samples == 0:
            return 0.0
        
        avg_duration = total_duration / valid_samples
        total_hours = (avg_duration * len(video_files)) / 3600
        
        return total_hours
    
    def _create_processing_batches(self, video_files: List[Path]) -> List[List[Path]]:
        """Create processing batches based on size limits."""
        batches = []
        current_batch = []
        current_size_gb = 0.0
        
        for video_file in video_files:
            file_size_gb = video_file.stat().st_size / (1024**3)
            
            if current_size_gb + file_size_gb > self.config.chunk_size_gb and current_batch:
                batches.append(current_batch)
                current_batch = [video_file]
                current_size_gb = file_size_gb
            else:
                current_batch.append(video_file)
                current_size_gb += file_size_gb
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _process_batch(
        self, 
        batch: List[Path], 
        dataset_id: str, 
        dataset_source: str,
        batch_idx: int
    ) -> List[str]:
        """Process a batch of video files in parallel."""
        
        # Allocate storage for batch
        batch_size_gb = sum(f.stat().st_size for f in batch) / (1024**3)
        storage_tier = self.storage_manager.allocate_storage(batch_size_gb, "medium")
        
        # Create batch output directory
        batch_output_dir = Path(storage_tier.path) / dataset_id / f"batch_{batch_idx:04d}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos in parallel
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit jobs
            futures = []
            for video_file in batch:
                future = executor.submit(
                    self._process_single_video,
                    str(video_file),
                    str(batch_output_dir),
                    dataset_id
                )
                futures.append((future, video_file))
            
            # Collect results
            output_paths = []
            for future, video_file in tqdm(futures, desc=f"Processing batch {batch_idx}"):
                try:
                    result = future.result()
                    if result['success']:
                        output_paths.extend(result['output_files'])
                        self.processed_count += 1
                        self.total_size_processed += video_file.stat().st_size / (1024**3)
                    else:
                        self.failed_count += 1
                        self.logger.error(f"Failed to process {video_file}: {result['error']}")
                except Exception as e:
                    self.failed_count += 1
                    self.logger.error(f"Exception processing {video_file}: {e}")
        
        return output_paths
    
    def _process_single_video(self, video_path: str, output_dir: str, dataset_id: str) -> Dict:
        """Process a single video file."""
        try:
            video_path = Path(video_path)
            output_dir = Path(output_dir)
            
            # Generate video ID
            video_id = self._generate_video_id(video_path)
            
            # Quality assessment
            quality_score = self._assess_video_quality(video_path)
            
            if quality_score < self.config.min_quality_score:
                return {
                    'success': False,
                    'error': f'Quality score {quality_score:.3f} below threshold {self.config.min_quality_score}'
                }
            
            # Video chunking with overlap
            chunk_config = ChunkConfig(
                chunk_duration=300,  # 5 minutes
                overlap_duration=30,  # 30 seconds
                use_scene_detection=True,
                sampling_strategy="adaptive"
            )
            
            chunker = VideoChunker(chunk_config)
            
            # Create video-specific output directory
            video_output_dir = output_dir / video_id
            video_output_dir.mkdir(exist_ok=True)
            
            # Process video chunks
            chunks = chunker.chunk_video(
                str(video_path),
                str(video_output_dir),
                cache_dir=str(video_output_dir / "cache")
            )
            
            # Deduplication
            if self.config.enable_deduplication:
                chunks = self._deduplicate_chunks(chunks)
            
            # Update database
            if self.database:
                file_size_gb = video_path.stat().st_size / (1024**3)
                duration = sum(chunk['duration'] for chunk in chunks)
                
                self.database.insert_video_file(
                    video_id, dataset_id, str(video_path),
                    file_size_gb, duration, quality_score
                )
                
                chunk_ids = [chunk['name'] for chunk in chunks]
                self.database.update_processing_status(video_id, "completed", chunk_ids)
            
            # Collect output files
            output_files = [chunk['path'] for chunk in chunks]
            
            return {
                'success': True,
                'video_id': video_id,
                'output_files': output_files,
                'quality_score': quality_score,
                'num_chunks': len(chunks)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _assess_video_quality(self, video_path: Path) -> float:
        """Assess video quality for filtering."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            # Sample frames for quality assessment
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_indices = np.linspace(0, frame_count-1, min(10, frame_count), dtype=int)
            
            sample_frames = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    sample_frames.append(frame)
            
            cap.release()
            
            if not sample_frames:
                return 0.0
            
            # Use quality filter to assess
            quality_score = self.quality_filter.assess_quality(sample_frames)
            return quality_score
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed for {video_path}: {e}")
            return 0.0
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on content hashing."""
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            try:
                # Load chunk data
                chunk_data = np.load(chunk['path'])
                frames = chunk_data['frames']
                
                # Compute content hash
                content_hash = hashlib.md5(frames.tobytes()).hexdigest()
                
                if content_hash not in seen_hashes:
                    unique_chunks.append(chunk)
                    seen_hashes.add(content_hash)
                else:
                    # Remove duplicate file
                    os.remove(chunk['path'])
                    self.logger.info(f"Removed duplicate chunk: {chunk['name']}")
            
            except Exception as e:
                self.logger.warning(f"Deduplication failed for chunk {chunk['name']}: {e}")
                unique_chunks.append(chunk)  # Keep on error
        
        return unique_chunks
    
    def _cleanup_memory(self):
        """Force memory cleanup between batches."""
        import gc
        gc.collect()
        
        # Log memory usage
        memory_usage = psutil.virtual_memory()
        self.logger.info(f"Memory usage: {memory_usage.percent:.1f}% ({memory_usage.used / (1024**3):.1f}GB)")
    
    def _generate_dataset_id(self, source: str, name: str) -> str:
        """Generate unique dataset ID."""
        timestamp = int(time.time())
        content = f"{source}_{name}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_video_id(self, video_path: Path) -> str:
        """Generate unique video ID."""
        content = f"{video_path.name}_{video_path.stat().st_size}_{video_path.stat().st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _finalize_dataset_metadata(self, dataset_id: str, source: str, output_paths: List[str]):
        """Create final dataset metadata."""
        if not self.database:
            return
        
        # Calculate final statistics
        total_size_gb = sum(Path(p).stat().st_size for p in output_paths if Path(p).exists()) / (1024**3)
        
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            source=source,
            total_hours=self.total_size_processed * 0.1,  # Rough estimate
            total_size_gb=total_size_gb,
            fps=10.0,  # Target FPS
            resolution=(256, 256),  # Processed resolution
            num_sequences=len(output_paths),
            quality_score=0.8,  # Average quality
            processing_date=time.strftime("%Y-%m-%d"),
            storage_tier="ssd_warm"
        )
        
        self.database.insert_dataset(metadata)
    
    def _generate_processing_report(self, dataset_id: str) -> str:
        """Generate processing report."""
        report = {
            'dataset_id': dataset_id,
            'processing_completed': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_videos_processed': self.processed_count,
            'total_videos_failed': self.failed_count,
            'total_size_processed_gb': self.total_size_processed,
            'success_rate': self.processed_count / max(1, self.processed_count + self.failed_count),
        }
        
        if self.database:
            stats = self.database.get_dataset_stats()
            report.update(stats)
        
        report_path = f"processing_report_{dataset_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
    
    def get_processing_status(self) -> Dict:
        """Get current processing status."""
        return {
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'total_size_processed_gb': self.total_size_processed,
            'memory_usage_gb': psutil.virtual_memory().used / (1024**3),
            'cpu_usage_percent': psutil.cpu_percent(),
        }


def create_large_scale_config() -> ProcessingConfig:
    """Create configuration for large-scale processing."""
    return ProcessingConfig(
        num_workers=max(4, mp.cpu_count() // 2),
        max_concurrent_videos=50,
        chunk_size_gb=20.0,
        min_quality_score=0.6,
        organize_by_quality=True,
        organize_by_scenario=True,
        enable_deduplication=True,
        enable_compression=True
    )


def main():
    """Example usage for large-scale processing."""
    config = create_large_scale_config()
    processor = LargeScaleProcessor(config)
    
    # Process dataset
    input_directory = "/path/to/100k_hour_dataset"
    report_path = processor.process_dataset_directory(input_directory, "waymo_100k")
    
    print(f"Processing completed. Report: {report_path}")
    
    # Show final statistics
    status = processor.get_processing_status()
    print(f"Final status: {status}")


if __name__ == "__main__":
    main()