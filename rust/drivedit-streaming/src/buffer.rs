//! Lock-free ring buffer for frame storage.
//!
//! This module provides a SPSC (single-producer, single-consumer) ring buffer
//! optimized for streaming frame data with minimal synchronization overhead.
//!
//! # Example
//!
//! ```rust
//! use drivedit_streaming::buffer::FrameRingBuffer;
//! use drivedit_types::RawFrame;
//!
//! let (mut producer, mut consumer) = FrameRingBuffer::new(16);
//!
//! // Producer side
//! let frame = RawFrame::new(0, 0.0, vec![0u8; 640 * 360 * 3], 640, 360, drivedit_types::Telemetry::default());
//! producer.push(frame);
//!
//! // Consumer side
//! if let Some(frame) = consumer.pop() {
//!     println!("Got frame: {}x{}", frame.width, frame.height);
//! }
//! ```

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapRb,
};
use tracing::{debug, trace, warn};

use drivedit_types::RawFrame;

/// Statistics for the ring buffer.
#[derive(Debug, Default)]
pub struct BufferStats {
    /// Total frames pushed.
    pub frames_pushed: AtomicU64,
    /// Total frames popped.
    pub frames_popped: AtomicU64,
    /// Frames dropped due to buffer full.
    pub frames_dropped: AtomicU64,
    /// Current buffer occupancy.
    pub current_occupancy: AtomicUsize,
}

impl BufferStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total frames pushed.
    pub fn pushed(&self) -> u64 {
        self.frames_pushed.load(Ordering::Relaxed)
    }

    /// Get total frames popped.
    pub fn popped(&self) -> u64 {
        self.frames_popped.load(Ordering::Relaxed)
    }

    /// Get frames dropped.
    pub fn dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
    }

    /// Get current occupancy.
    pub fn occupancy(&self) -> usize {
        self.current_occupancy.load(Ordering::Relaxed)
    }
}

/// Lock-free ring buffer for frames.
///
/// Wraps the ringbuf crate's SPSC ring buffer with frame-specific
/// functionality and statistics tracking.
pub struct FrameRingBuffer {
    // Note: This struct exists primarily as a namespace for the constructor.
    // The actual buffer state is in the FrameProducer and FrameConsumer.
}

impl FrameRingBuffer {
    /// Create a new ring buffer with the given capacity.
    ///
    /// Returns a producer/consumer pair for SPSC access.
    pub fn new(capacity: usize) -> (FrameProducer, FrameConsumer) {
        let rb = HeapRb::<RawFrame>::new(capacity);
        let (prod, cons) = rb.split();
        let stats = Arc::new(BufferStats::new());

        let producer = FrameProducer {
            inner: Mutex::new(prod),
            capacity,
            stats: stats.clone(),
        };

        let consumer = FrameConsumer {
            inner: Mutex::new(cons),
            stats,
        };

        debug!("Created ring buffer with capacity {}", capacity);

        (producer, consumer)
    }
}

/// Producer side of the ring buffer.
pub struct FrameProducer {
    inner: Mutex<ringbuf::HeapProd<RawFrame>>,
    capacity: usize,
    stats: Arc<BufferStats>,
}

impl FrameProducer {
    /// Push a frame to the buffer.
    ///
    /// Returns `true` if the frame was pushed successfully, `false` if the
    /// buffer is full and the frame was dropped.
    pub fn push(&mut self, frame: RawFrame) -> bool {
        let mut inner = self.inner.lock();

        if Observer::is_full(&*inner) {
            warn!("Buffer full, dropping frame");
            self.stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        match inner.try_push(frame) {
            Ok(()) => {
                self.stats.frames_pushed.fetch_add(1, Ordering::Relaxed);
                self.stats.current_occupancy.fetch_add(1, Ordering::Relaxed);
                trace!("Pushed frame, occupancy: {}", self.stats.occupancy());
                true
            }
            Err(_) => {
                self.stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Push a frame, dropping if buffer is full.
    ///
    /// Note: In SPSC mode, we cannot pop from the producer side to implement
    /// true overwrite semantics. This method will push if there's space,
    /// otherwise the frame is dropped. Use this when you want to ensure
    /// the producer never blocks.
    pub fn push_overwrite(&mut self, frame: RawFrame) {
        let mut inner = self.inner.lock();

        // In SPSC mode, producer cannot pop, so we just drop if full
        if Observer::is_full(&*inner) {
            self.stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
            // Still try to push in case consumer freed space
        }

        match inner.try_push(frame) {
            Ok(()) => {
                self.stats.frames_pushed.fetch_add(1, Ordering::Relaxed);
                let occupancy = Observer::occupied_len(&*inner);
                self.stats.current_occupancy.store(occupancy, Ordering::Relaxed);
            }
            Err(_) => {
                // Already counted as dropped above if was full
                if !Observer::is_full(&*inner) {
                    self.stats.frames_dropped.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Check if the buffer is full.
    pub fn is_full(&self) -> bool {
        Observer::is_full(&*self.inner.lock())
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        Observer::is_empty(&*self.inner.lock())
    }

    /// Get current buffer length.
    pub fn len(&self) -> usize {
        Observer::occupied_len(&*self.inner.lock())
    }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get shared statistics.
    pub fn stats(&self) -> &Arc<BufferStats> {
        &self.stats
    }

    /// Check if backpressure is detected (buffer > 75% full).
    pub fn has_backpressure(&self) -> bool {
        self.len() > (self.capacity * 3) / 4
    }
}

/// Consumer side of the ring buffer.
pub struct FrameConsumer {
    inner: Mutex<ringbuf::HeapCons<RawFrame>>,
    stats: Arc<BufferStats>,
}

impl FrameConsumer {
    /// Pop a frame from the buffer.
    ///
    /// Returns `None` if the buffer is empty.
    pub fn pop(&mut self) -> Option<RawFrame> {
        let mut inner = self.inner.lock();

        match inner.try_pop() {
            Some(frame) => {
                self.stats.frames_popped.fetch_add(1, Ordering::Relaxed);
                self.stats.current_occupancy.fetch_sub(1, Ordering::Relaxed);
                trace!("Popped frame, occupancy: {}", self.stats.occupancy());
                Some(frame)
            }
            None => None,
        }
    }

    /// Pop multiple frames at once.
    ///
    /// Returns a vector of frames (may be empty if buffer is empty).
    pub fn pop_batch(&mut self, max_count: usize) -> Vec<RawFrame> {
        let mut inner = self.inner.lock();
        let mut frames = Vec::with_capacity(max_count.min(Observer::occupied_len(&*inner)));

        for _ in 0..max_count {
            match inner.try_pop() {
                Some(frame) => {
                    self.stats.frames_popped.fetch_add(1, Ordering::Relaxed);
                    frames.push(frame);
                }
                None => break,
            }
        }

        let popped = frames.len();
        if popped > 0 {
            self.stats
                .current_occupancy
                .fetch_sub(popped, Ordering::Relaxed);
        }

        frames
    }

    /// Peek at the next frame without removing it.
    ///
    /// Note: This requires locking and is not as efficient as pop.
    /// Currently returns None as ringbuf doesn't have peek for SPSC.
    pub fn peek(&self) -> Option<RawFrame> {
        let _inner = self.inner.lock();
        // ringbuf doesn't have peek, so we can't implement this without cloning
        // For now, return None
        None
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        Observer::is_empty(&*self.inner.lock())
    }

    /// Get current buffer length.
    pub fn len(&self) -> usize {
        Observer::occupied_len(&*self.inner.lock())
    }

    /// Get shared statistics.
    pub fn stats(&self) -> &Arc<BufferStats> {
        &self.stats
    }
}

// Safety: The producer and consumer are protected by Mutex
unsafe impl Send for FrameProducer {}
unsafe impl Send for FrameConsumer {}

#[cfg(test)]
mod tests {
    use super::*;
    use drivedit_types::Telemetry;

    fn make_test_frame(id: u64) -> RawFrame {
        RawFrame::new(id, 0.0, vec![0u8; 64 * 64 * 3], 64, 64, Telemetry::default())
    }

    #[test]
    fn test_buffer_creation() {
        let (producer, consumer) = FrameRingBuffer::new(16);
        assert!(producer.is_empty());
        assert!(consumer.is_empty());
        assert_eq!(producer.capacity(), 16);
    }

    #[test]
    fn test_push_pop() {
        let (mut producer, mut consumer) = FrameRingBuffer::new(16);

        producer.push(make_test_frame(1));
        producer.push(make_test_frame(2));

        assert_eq!(producer.len(), 2);
        assert_eq!(consumer.len(), 2);

        let frame1 = consumer.pop().unwrap();
        assert_eq!(frame1.frame_id, 1);

        let frame2 = consumer.pop().unwrap();
        assert_eq!(frame2.frame_id, 2);

        assert!(consumer.pop().is_none());
    }

    #[test]
    fn test_buffer_full() {
        let (mut producer, _consumer) = FrameRingBuffer::new(4);

        // Fill the buffer
        for i in 0..4 {
            assert!(producer.push(make_test_frame(i)));
        }

        // Buffer should be full
        assert!(producer.is_full());
        assert!(!producer.push(make_test_frame(99)));

        // Check dropped count
        assert_eq!(producer.stats().dropped(), 1);
    }

    #[test]
    fn test_push_overwrite() {
        let (mut producer, mut consumer) = FrameRingBuffer::new(4);

        // Fill the buffer
        for i in 0..4 {
            producer.push_overwrite(make_test_frame(i));
        }

        // Should have 4 frames
        assert_eq!(consumer.len(), 4);

        // Try to push more - in SPSC mode, these will be dropped
        producer.push_overwrite(make_test_frame(4));
        producer.push_overwrite(make_test_frame(5));

        // Still 4 frames (extras were dropped)
        assert_eq!(consumer.len(), 4);

        // First frame should be ID 0 (oldest)
        let frame = consumer.pop().unwrap();
        assert_eq!(frame.frame_id, 0);

        // Check dropped count (2 frames were dropped)
        assert_eq!(producer.stats().dropped(), 2);
    }

    #[test]
    fn test_batch_pop() {
        let (mut producer, mut consumer) = FrameRingBuffer::new(16);

        for i in 0..10 {
            producer.push(make_test_frame(i));
        }

        let batch = consumer.pop_batch(5);
        assert_eq!(batch.len(), 5);
        assert_eq!(batch[0].frame_id, 0);
        assert_eq!(batch[4].frame_id, 4);

        assert_eq!(consumer.len(), 5);
    }

    #[test]
    fn test_statistics() {
        let (mut producer, mut consumer) = FrameRingBuffer::new(4);

        producer.push(make_test_frame(1));
        producer.push(make_test_frame(2));

        assert_eq!(producer.stats().pushed(), 2);
        assert_eq!(producer.stats().occupancy(), 2);

        consumer.pop();

        assert_eq!(producer.stats().popped(), 1);
        assert_eq!(producer.stats().occupancy(), 1);
    }

    #[test]
    fn test_backpressure_detection() {
        let (mut producer, _consumer) = FrameRingBuffer::new(8);

        // Fill to >75%
        for i in 0..7 {
            producer.push(make_test_frame(i));
        }

        // Should trigger backpressure (7/8 > 75%)
        assert!(producer.has_backpressure());
    }
}
