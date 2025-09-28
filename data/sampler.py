"""
Data sampling utilities for temporal and balanced sampling.
Handles various sampling strategies for video sequences.
"""

import torch
import torch.utils.data as data
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import random
import math
import numpy as np
from collections import defaultdict, Counter


class TemporalSampler(data.Sampler):
    """Sampler for temporal sequences with various strategies."""
    
    def __init__(
        self,
        dataset,
        sampling_strategy: str = 'uniform',
        sequence_length: int = 16,
        temporal_stride: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize temporal sampler.
        
        Args:
            dataset: Dataset to sample from
            sampling_strategy: Sampling strategy ('uniform', 'dense', 'sparse', 'adaptive')
            sequence_length: Length of sequences to sample
            temporal_stride: Stride for temporal sampling
            shuffle: Whether to shuffle samples
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.sampling_strategy = sampling_strategy
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.shuffle = shuffle
        
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None
        
        # Build temporal indices
        self.temporal_indices = self._build_temporal_indices()
    
    def _build_temporal_indices(self) -> List[Tuple[int, int]]:
        """Build temporal sampling indices."""
        indices = []
        
        if hasattr(self.dataset, 'sequences'):
            # Dataset with sequence information
            for seq_idx, sequence in enumerate(self.dataset.sequences):
                num_frames = sequence.get('num_frames', len(sequence.get('frame_paths', [])))
                sequence_indices = self._sample_from_sequence(num_frames, seq_idx)
                indices.extend(sequence_indices)
        else:
            # Fallback: treat as single long sequence
            total_length = len(self.dataset)
            sequence_indices = self._sample_from_sequence(total_length, 0)
            indices.extend(sequence_indices)
        
        return indices
    
    def _sample_from_sequence(self, num_frames: int, seq_idx: int) -> List[Tuple[int, int]]:
        """Sample indices from a single sequence."""
        indices = []
        
        if self.sampling_strategy == 'uniform':
            # Uniform sampling across sequence
            max_start = num_frames - (self.sequence_length - 1) * self.temporal_stride - 1
            if max_start > 0:
                num_samples = max(1, max_start // (self.sequence_length // 2))
                for i in range(num_samples):
                    start_idx = i * (self.sequence_length // 2)
                    if start_idx <= max_start:
                        indices.append((seq_idx, start_idx))
        
        elif self.sampling_strategy == 'dense':
            # Dense sampling (every frame as start)
            max_start = num_frames - (self.sequence_length - 1) * self.temporal_stride - 1
            for start_idx in range(0, max_start + 1):
                indices.append((seq_idx, start_idx))
        
        elif self.sampling_strategy == 'sparse':
            # Sparse sampling (larger gaps)
            max_start = num_frames - (self.sequence_length - 1) * self.temporal_stride - 1
            if max_start > 0:
                step = max(1, self.sequence_length)
                for start_idx in range(0, max_start + 1, step):
                    indices.append((seq_idx, start_idx))
        
        elif self.sampling_strategy == 'adaptive':
            # Adaptive sampling based on sequence length
            max_start = num_frames - (self.sequence_length - 1) * self.temporal_stride - 1
            if max_start > 0:
                if num_frames < self.sequence_length * 2:
                    # Short sequence: dense sampling
                    for start_idx in range(0, max_start + 1):
                        indices.append((seq_idx, start_idx))
                else:
                    # Long sequence: sparse sampling
                    num_samples = max(3, num_frames // self.sequence_length)
                    for i in range(num_samples):
                        start_idx = i * max_start // (num_samples - 1) if num_samples > 1 else 0
                        indices.append((seq_idx, min(start_idx, max_start)))
        
        return indices
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over sample indices."""
        indices = list(range(len(self.temporal_indices)))
        
        if self.shuffle:
            if self.generator is not None:
                g = torch.Generator()
                g.set_state(self.generator.get_state())
                indices = torch.randperm(len(indices), generator=g).tolist()
            else:
                random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.temporal_indices)


class BalancedSampler(data.Sampler):
    """Balanced sampler for handling class imbalance in video data."""
    
    def __init__(
        self,
        dataset,
        class_key: str = 'class_label',
        oversample_minority: bool = True,
        undersample_majority: bool = False,
        replacement: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset to sample from
            class_key: Key for class labels in dataset
            oversample_minority: Whether to oversample minority classes
            undersample_majority: Whether to undersample majority classes
            replacement: Whether to sample with replacement
            seed: Random seed
        """
        self.dataset = dataset
        self.class_key = class_key
        self.oversample_minority = oversample_minority
        self.undersample_majority = undersample_majority
        self.replacement = replacement
        
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None
        
        # Build class indices
        self.class_indices = self._build_class_indices()
        self.balanced_indices = self._create_balanced_indices()
    
    def _build_class_indices(self) -> Dict[Any, List[int]]:
        """Build mapping from classes to sample indices."""
        class_indices = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            try:
                if hasattr(self.dataset, 'get_sequence_info'):
                    # Get class from sequence info
                    info = self.dataset.get_sequence_info(idx)
                    class_label = info.get(self.class_key, 'unknown')
                else:
                    # Try to get class from sample
                    sample = self.dataset[idx]
                    class_label = sample.get(self.class_key, 'unknown')
                
                class_indices[class_label].append(idx)
            except:
                # Fallback: assign to default class
                class_indices['unknown'].append(idx)
        
        return dict(class_indices)
    
    def _create_balanced_indices(self) -> List[int]:
        """Create balanced sample indices."""
        if not self.class_indices:
            return list(range(len(self.dataset)))
        
        class_sizes = {cls: len(indices) for cls, indices in self.class_indices.items()}
        
        if self.oversample_minority and self.undersample_majority:
            # Balance to median size
            target_size = sorted(class_sizes.values())[len(class_sizes) // 2]
        elif self.oversample_minority:
            # Balance to largest class
            target_size = max(class_sizes.values())
        elif self.undersample_majority:
            # Balance to smallest class
            target_size = min(class_sizes.values())
        else:
            # No balancing
            return list(range(len(self.dataset)))
        
        balanced_indices = []
        
        for class_label, indices in self.class_indices.items():
            current_size = len(indices)
            
            if current_size == target_size:
                balanced_indices.extend(indices)
            elif current_size < target_size:
                # Oversample
                if self.replacement:
                    # Sample with replacement
                    if self.generator is not None:
                        sampled_indices = torch.multinomial(
                            torch.ones(current_size),
                            target_size,
                            replacement=True,
                            generator=self.generator
                        ).tolist()
                    else:
                        sampled_indices = np.random.choice(current_size, target_size, replace=True).tolist()
                    
                    balanced_indices.extend([indices[i] for i in sampled_indices])
                else:
                    # Repeat indices cyclically
                    repeats = target_size // current_size
                    remainder = target_size % current_size
                    
                    balanced_indices.extend(indices * repeats)
                    if remainder > 0:
                        balanced_indices.extend(indices[:remainder])
            else:
                # Undersample
                if self.generator is not None:
                    sampled_indices = torch.randperm(current_size, generator=self.generator)[:target_size].tolist()
                else:
                    sampled_indices = np.random.permutation(current_size)[:target_size].tolist()
                
                balanced_indices.extend([indices[i] for i in sampled_indices])
        
        return balanced_indices
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over balanced sample indices."""
        if self.generator is not None:
            g = torch.Generator()
            g.set_state(self.generator.get_state())
            indices = torch.randperm(len(self.balanced_indices), generator=g).tolist()
        else:
            indices = np.random.permutation(len(self.balanced_indices)).tolist()
        
        return iter([self.balanced_indices[i] for i in indices])
    
    def __len__(self) -> int:
        return len(self.balanced_indices)
    
    def get_class_distribution(self) -> Dict[Any, int]:
        """Get distribution of classes in balanced sampling."""
        class_counts = Counter()
        for idx in self.balanced_indices:
            # Find class for this index
            for class_label, indices in self.class_indices.items():
                if idx in indices:
                    class_counts[class_label] += 1
                    break
        return dict(class_counts)


class ChunkedSampler(data.Sampler):
    """Sampler that processes sequences in chunks for memory efficiency."""
    
    def __init__(
        self,
        dataset,
        chunk_size: int = 32,
        overlap: int = 4,
        shuffle_chunks: bool = True,
        shuffle_within_chunks: bool = False,
        drop_incomplete: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize chunked sampler.
        
        Args:
            dataset: Dataset to sample from
            chunk_size: Size of each chunk
            overlap: Overlap between consecutive chunks
            shuffle_chunks: Whether to shuffle chunk order
            shuffle_within_chunks: Whether to shuffle within chunks
            drop_incomplete: Whether to drop incomplete chunks
            seed: Random seed
        """
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunks = shuffle_within_chunks
        self.drop_incomplete = drop_incomplete
        
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
            random.seed(seed)
        else:
            self.generator = None
        
        self.chunks = self._create_chunks()
    
    def _create_chunks(self) -> List[List[int]]:
        """Create chunks from dataset indices."""
        total_samples = len(self.dataset)
        chunks = []
        
        step = self.chunk_size - self.overlap
        
        for start_idx in range(0, total_samples, step):
            end_idx = min(start_idx + self.chunk_size, total_samples)
            
            chunk = list(range(start_idx, end_idx))
            
            if len(chunk) == self.chunk_size or not self.drop_incomplete:
                chunks.append(chunk)
        
        return chunks
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over chunked samples."""
        chunks = self.chunks.copy()
        
        # Shuffle chunks if requested
        if self.shuffle_chunks:
            if self.generator is not None:
                g = torch.Generator()
                g.set_state(self.generator.get_state())
                chunk_order = torch.randperm(len(chunks), generator=g).tolist()
                chunks = [chunks[i] for i in chunk_order]
            else:
                random.shuffle(chunks)
        
        # Process chunks
        for chunk in chunks:
            if self.shuffle_within_chunks:
                if self.generator is not None:
                    g = torch.Generator()
                    g.set_state(self.generator.get_state())
                    chunk_order = torch.randperm(len(chunk), generator=g).tolist()
                    chunk = [chunk[i] for i in chunk_order]
                else:
                    random.shuffle(chunk)
            
            for idx in chunk:
                yield idx
    
    def __len__(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)
    
    def get_chunk_info(self) -> Dict[str, Any]:
        """Get information about chunks."""
        chunk_sizes = [len(chunk) for chunk in self.chunks]
        return {
            'num_chunks': len(self.chunks),
            'chunk_sizes': chunk_sizes,
            'avg_chunk_size': np.mean(chunk_sizes),
            'total_samples': sum(chunk_sizes)
        }


class SequentialChunkSampler(data.Sampler):
    """Sequential sampler that maintains temporal order within chunks."""
    
    def __init__(
        self,
        dataset,
        sequences_per_chunk: int = 8,
        maintain_sequence_order: bool = True,
        shuffle_sequences: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize sequential chunk sampler.
        
        Args:
            dataset: Dataset to sample from
            sequences_per_chunk: Number of sequences per chunk
            maintain_sequence_order: Whether to maintain temporal order within sequences
            shuffle_sequences: Whether to shuffle sequence order
            seed: Random seed
        """
        self.dataset = dataset
        self.sequences_per_chunk = sequences_per_chunk
        self.maintain_sequence_order = maintain_sequence_order
        self.shuffle_sequences = shuffle_sequences
        
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
            random.seed(seed)
        else:
            self.generator = None
        
        self.sequence_chunks = self._create_sequence_chunks()
    
    def _create_sequence_chunks(self) -> List[List[int]]:
        """Create chunks organized by sequences."""
        if hasattr(self.dataset, 'sequence_indices'):
            # Group by sequence
            sequence_groups = defaultdict(list)
            for idx, (seq_idx, start_idx) in enumerate(self.dataset.sequence_indices):
                sequence_groups[seq_idx].append(idx)
            
            # Create chunks
            chunks = []
            current_chunk = []
            
            for seq_idx, indices in sequence_groups.items():
                if self.maintain_sequence_order:
                    indices = sorted(indices, key=lambda i: self.dataset.sequence_indices[i][1])
                
                current_chunk.extend(indices)
                
                if len(current_chunk) >= self.sequences_per_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            
            # Add remaining samples
            if current_chunk:
                chunks.append(current_chunk)
        
        else:
            # Fallback: regular chunking
            total_samples = len(self.dataset)
            chunks = []
            
            for start_idx in range(0, total_samples, self.sequences_per_chunk):
                end_idx = min(start_idx + self.sequences_per_chunk, total_samples)
                chunks.append(list(range(start_idx, end_idx)))
        
        return chunks
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over sequential chunks."""
        chunks = self.sequence_chunks.copy()
        
        # Shuffle chunk order if requested
        if self.shuffle_sequences:
            if self.generator is not None:
                g = torch.Generator()
                g.set_state(self.generator.get_state())
                chunk_order = torch.randperm(len(chunks), generator=g).tolist()
                chunks = [chunks[i] for i in chunk_order]
            else:
                random.shuffle(chunks)
        
        # Yield samples maintaining order within chunks
        for chunk in chunks:
            for idx in chunk:
                yield idx
    
    def __len__(self) -> int:
        return sum(len(chunk) for chunk in self.sequence_chunks)


class AdaptiveSampler(data.Sampler):
    """Adaptive sampler that adjusts based on training progress."""
    
    def __init__(
        self,
        dataset,
        initial_strategy: str = 'uniform',
        adaptation_frequency: int = 1000,
        difficulty_key: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize adaptive sampler.
        
        Args:
            dataset: Dataset to sample from
            initial_strategy: Initial sampling strategy
            adaptation_frequency: How often to adapt sampling
            difficulty_key: Key for sample difficulty scores
            seed: Random seed
        """
        self.dataset = dataset
        self.current_strategy = initial_strategy
        self.adaptation_frequency = adaptation_frequency
        self.difficulty_key = difficulty_key
        
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None
        
        self.step_count = 0
        self.difficulty_scores = {}
        
        # Initialize base sampler
        self.base_sampler = self._create_base_sampler()
    
    def _create_base_sampler(self) -> data.Sampler:
        """Create base sampler based on current strategy."""
        if self.current_strategy == 'uniform':
            return data.RandomSampler(self.dataset, generator=self.generator)
        elif self.current_strategy == 'balanced':
            return BalancedSampler(self.dataset, seed=self.generator.initial_seed() if self.generator else None)
        elif self.current_strategy == 'temporal':
            return TemporalSampler(self.dataset, seed=self.generator.initial_seed() if self.generator else None)
        else:
            return data.RandomSampler(self.dataset, generator=self.generator)
    
    def update_difficulty_scores(self, sample_indices: List[int], scores: List[float]) -> None:
        """Update difficulty scores for samples."""
        for idx, score in zip(sample_indices, scores):
            self.difficulty_scores[idx] = score
    
    def adapt_strategy(self) -> None:
        """Adapt sampling strategy based on current state."""
        if len(self.difficulty_scores) < 100:  # Need enough samples
            return
        
        # Analyze difficulty distribution
        scores = list(self.difficulty_scores.values())
        mean_difficulty = np.mean(scores)
        std_difficulty = np.std(scores)
        
        # Adapt strategy based on difficulty
        if std_difficulty > 0.3:  # High variance in difficulty
            if self.current_strategy != 'balanced':
                self.current_strategy = 'balanced'
                self.base_sampler = self._create_base_sampler()
        elif mean_difficulty > 0.7:  # Generally difficult
            if self.current_strategy != 'temporal':
                self.current_strategy = 'temporal'
                self.base_sampler = self._create_base_sampler()
        else:  # Balanced difficulty
            if self.current_strategy != 'uniform':
                self.current_strategy = 'uniform'
                self.base_sampler = self._create_base_sampler()
    
    def __iter__(self) -> Iterator[int]:
        """Iterate with adaptive sampling."""
        # Check if adaptation is needed
        if self.step_count % self.adaptation_frequency == 0:
            self.adapt_strategy()
        
        self.step_count += 1
        
        return iter(self.base_sampler)
    
    def __len__(self) -> int:
        return len(self.base_sampler)
    
    def get_current_strategy(self) -> str:
        """Get current sampling strategy."""
        return self.current_strategy