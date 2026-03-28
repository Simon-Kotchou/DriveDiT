// Suppress dead code warnings for simulation types used in benchmarks
#![allow(dead_code)]

//! Streaming latency benchmarks for DriveDiT data pipeline.
//!
//! Performance targets (from ENGINEERING_PLAN.md):
//! - Ring buffer push/pop: < 1us
//! - Frame-telemetry sync latency: < 100us
//! - End-to-end pipeline latency: < 5ms
//! - Throughput: > 100 fps sustained
//!
//! Run with: cargo bench --bench streaming_latency

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use parking_lot::{Condvar, Mutex};
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapRb,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Simulated Types (mirroring drivedit-streaming)
// ============================================================================

/// Vehicle telemetry.
#[derive(Debug, Clone, Default)]
struct Telemetry {
    pos: [f32; 3],
    forward: [f32; 3],
    up: [f32; 3],
    right: [f32; 3],
    speed_kmh: f32,
    steering: f32,
    throttle: f32,
    brake: f32,
    acceleration: f32,
    distance_total: f32,
    heading_rad: f32,
    heading_rate: f32,
}

impl Telemetry {
    fn to_array(&self) -> [f32; 20] {
        [
            self.pos[0], self.pos[1], self.pos[2],
            self.forward[0], self.forward[1], self.forward[2],
            self.up[0], self.up[1], self.up[2],
            self.right[0], self.right[1], self.right[2],
            self.speed_kmh, self.steering, self.throttle, self.brake,
            self.acceleration, self.distance_total, self.heading_rad, self.heading_rate,
        ]
    }
}

/// Single captured frame.
#[derive(Clone)]
struct Frame {
    frame_id: u64,
    timestamp_ms: f64,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    telemetry: Telemetry,
}

impl Frame {
    fn new(
        frame_id: u64,
        timestamp_ms: f64,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        telemetry: Telemetry,
    ) -> Self {
        Self {
            frame_id,
            timestamp_ms,
            pixels,
            width,
            height,
            telemetry,
        }
    }
}

/// Batch of frames.
struct FrameBatch {
    pixels: Vec<u8>,
    telemetry: Vec<f32>,
    frame_ids: Vec<u64>,
    timestamps: Vec<f64>,
    batch_size: usize,
    width: u32,
    height: u32,
}

impl FrameBatch {
    fn with_capacity(capacity: usize, width: u32, height: u32) -> Self {
        let frame_size = (width as usize) * (height as usize) * 3;
        Self {
            pixels: Vec::with_capacity(capacity * frame_size),
            telemetry: Vec::with_capacity(capacity * 20),
            frame_ids: Vec::with_capacity(capacity),
            timestamps: Vec::with_capacity(capacity),
            batch_size: 0,
            width,
            height,
        }
    }

    fn push(&mut self, frame: Frame) {
        self.pixels.extend_from_slice(&frame.pixels);
        self.telemetry.extend_from_slice(&frame.telemetry.to_array());
        self.frame_ids.push(frame.frame_id);
        self.timestamps.push(frame.timestamp_ms);
        self.batch_size += 1;
    }

    fn is_empty(&self) -> bool {
        self.batch_size == 0
    }
}

// ============================================================================
// Test Helpers
// ============================================================================

fn create_test_frame(id: u64, width: u32, height: u32) -> Frame {
    let pixel_count = (width as usize) * (height as usize) * 3;
    let mut pixels = vec![0u8; pixel_count];

    // Fill with deterministic pattern based on frame_id
    for (i, pixel) in pixels.iter_mut().enumerate() {
        *pixel = ((id as usize + i) % 256) as u8;
    }

    Frame::new(
        id,
        id as f64 * 200.0,
        pixels,
        width,
        height,
        Telemetry::default(),
    )
}

fn create_lightweight_frame(id: u64) -> Frame {
    // Minimal frame for throughput testing
    Frame::new(
        id,
        id as f64 * 200.0,
        vec![0u8; 640 * 360 * 3],
        640,
        360,
        Telemetry::default(),
    )
}

// ============================================================================
// Ring Buffer Implementation (simplified from drivedit-streaming)
// ============================================================================

/// Buffer statistics.
#[repr(align(64))]
#[derive(Debug, Default)]
struct BufferStats {
    frames_pushed: AtomicU64,
    frames_popped: AtomicU64,
    current_size: AtomicUsize,
}

/// Notification mechanism.
struct Notifier {
    mutex: Mutex<()>,
    condvar: Condvar,
}

impl Notifier {
    fn new() -> Self {
        Self {
            mutex: Mutex::new(()),
            condvar: Condvar::new(),
        }
    }

    fn notify_one(&self) {
        let _guard = self.mutex.lock();
        self.condvar.notify_one();
    }

    fn wait_timeout(&self, timeout: Duration) -> bool {
        let mut guard = self.mutex.lock();
        let result = self.condvar.wait_for(&mut guard, timeout);
        !result.timed_out()
    }
}

/// Frame buffer using lock-free SPSC ring buffer.
struct FrameBuffer {
    capacity: usize,
    stats: Arc<BufferStats>,
}

impl FrameBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            stats: Arc::new(BufferStats::default()),
        }
    }

    fn split(self) -> (FrameProducer, FrameConsumer) {
        let rb = HeapRb::<Frame>::new(self.capacity);
        let (prod, cons) = rb.split();

        let notifier = Arc::new(Notifier::new());
        let producer_active = Arc::new(AtomicBool::new(true));

        let producer = FrameProducer {
            inner: prod,
            stats: Arc::clone(&self.stats),
            notifier: Arc::clone(&notifier),
            producer_active: Arc::clone(&producer_active),
        };

        let consumer = FrameConsumer {
            inner: cons,
            capacity: self.capacity,
            stats: self.stats,
            notifier,
            producer_active,
        };

        (producer, consumer)
    }
}

/// Producer side of frame buffer.
struct FrameProducer {
    inner: ringbuf::HeapProd<Frame>,
    stats: Arc<BufferStats>,
    notifier: Arc<Notifier>,
    producer_active: Arc<AtomicBool>,
}

impl FrameProducer {
    fn push(&mut self, frame: Frame) -> Result<(), ()> {
        if self.inner.is_full() {
            return Err(());
        }

        let _ = self.inner.try_push(frame);
        self.stats.frames_pushed.fetch_add(1, Ordering::Relaxed);
        self.stats.current_size.fetch_add(1, Ordering::Relaxed);
        self.notifier.notify_one();

        Ok(())
    }

    fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    fn len(&self) -> usize {
        self.inner.occupied_len()
    }
}

impl Drop for FrameProducer {
    fn drop(&mut self) {
        self.producer_active.store(false, Ordering::Release);
        self.notifier.notify_one();
    }
}

/// Consumer side of frame buffer.
struct FrameConsumer {
    inner: ringbuf::HeapCons<Frame>,
    capacity: usize,
    stats: Arc<BufferStats>,
    notifier: Arc<Notifier>,
    producer_active: Arc<AtomicBool>,
}

impl FrameConsumer {
    fn try_pop(&mut self) -> Option<Frame> {
        match self.inner.try_pop() {
            Some(frame) => {
                self.stats.frames_popped.fetch_add(1, Ordering::Relaxed);
                self.stats.current_size.fetch_sub(1, Ordering::Relaxed);
                Some(frame)
            }
            None => None,
        }
    }

    fn pop_batch(&mut self, count: usize, width: u32, height: u32) -> FrameBatch {
        let mut batch = FrameBatch::with_capacity(count, width, height);

        for _ in 0..count {
            match self.try_pop() {
                Some(frame) => batch.push(frame),
                None => break,
            }
        }

        batch
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn len(&self) -> usize {
        self.inner.occupied_len()
    }

    fn is_producer_active(&self) -> bool {
        self.producer_active.load(Ordering::Acquire)
    }
}

// ============================================================================
// Ring Buffer Benchmarks
// ============================================================================

/// Target: < 1us for push/pop
fn bench_ring_buffer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer");

    // Single push operation
    group.bench_function("push_single", |b| {
        let buffer = FrameBuffer::new(256);
        let (mut producer, mut consumer) = buffer.split();

        b.iter(|| {
            let frame = create_lightweight_frame(1);
            producer.push(black_box(frame)).ok();
            // Pop to prevent buffer from filling up
            consumer.try_pop();
        });
    });

    // Single pop operation
    group.bench_function("pop_single", |b| {
        let buffer = FrameBuffer::new(256);
        let (mut producer, mut consumer) = buffer.split();

        // Pre-fill buffer
        for i in 0..128 {
            producer.push(create_lightweight_frame(i)).unwrap();
        }

        b.iter(|| {
            if consumer.is_empty() {
                // Refill when empty
                for i in 0..128 {
                    producer.push(create_lightweight_frame(i)).unwrap();
                }
            }
            black_box(consumer.try_pop())
        });
    });

    // Push + pop combined (typical streaming pattern)
    group.bench_function("push_pop_combined", |b| {
        let buffer = FrameBuffer::new(256);
        let (mut producer, mut consumer) = buffer.split();

        // Pre-fill halfway
        for i in 0..64 {
            producer.push(create_lightweight_frame(i)).unwrap();
        }

        let mut id = 64u64;
        b.iter(|| {
            producer.push(create_lightweight_frame(id)).ok();
            id += 1;
            black_box(consumer.try_pop())
        });
    });

    // Batch pop operation
    group.bench_function("pop_batch_8", |b| {
        let buffer = FrameBuffer::new(256);
        let (mut producer, mut consumer) = buffer.split();

        b.iter(|| {
            // Refill if needed
            while consumer.len() < 8 {
                producer.push(create_lightweight_frame(0)).ok();
            }
            black_box(consumer.pop_batch(8, 640, 360))
        });
    });

    group.finish();
}

/// Throughput benchmark: sustained push/pop rate
fn bench_ring_buffer_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer_throughput");
    group.throughput(Throughput::Elements(1000));

    group.bench_function("push_pop_1000", |b| {
        b.iter(|| {
            let buffer = FrameBuffer::new(256);
            let (mut producer, mut consumer) = buffer.split();

            for i in 0..1000u64 {
                let frame = Frame::new(
                    i,
                    i as f64 * 200.0,
                    vec![0u8; 640 * 360 * 3],
                    640,
                    360,
                    Telemetry::default(),
                );

                producer.push(frame).ok();
                if i % 2 == 1 {
                    consumer.try_pop();
                }
            }

            // Drain remaining
            while !consumer.is_empty() {
                consumer.try_pop();
            }
        });
    });

    // Concurrent producer-consumer throughput
    group.bench_function("concurrent_1000", |b| {
        b.iter(|| {
            let buffer = FrameBuffer::new(256);
            let (mut producer, mut consumer) = buffer.split();

            // Producer thread
            let producer_handle = thread::spawn(move || {
                for i in 0..1000u64 {
                    let frame = create_lightweight_frame(i);
                    // Retry on full
                    while producer.push(frame.clone()).is_err() {
                        thread::yield_now();
                    }
                }
            });

            // Consumer in main thread
            let mut received = 0;
            while received < 1000 {
                if let Some(_) = consumer.try_pop() {
                    received += 1;
                } else if !consumer.is_producer_active() {
                    break;
                } else {
                    thread::yield_now();
                }
            }

            producer_handle.join().unwrap();
            black_box(received)
        });
    });

    group.finish();
}

// ============================================================================
// Frame-Telemetry Sync Benchmarks
// ============================================================================

/// Pending frame data waiting for telemetry.
struct PendingFrame {
    frame_id: u64,
    timestamp_ms: f64,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    added_at: Instant,
}

/// Pending telemetry waiting for frame.
struct PendingTelemetry {
    telemetry: Telemetry,
    added_at: Instant,
}

/// Frame-telemetry synchronizer.
struct FrameTelemetrySync {
    pending_frames: HashMap<u64, PendingFrame>,
    pending_telemetry: HashMap<u64, PendingTelemetry>,
    matched: u64,
}

impl FrameTelemetrySync {
    fn new() -> Self {
        Self {
            pending_frames: HashMap::with_capacity(64),
            pending_telemetry: HashMap::with_capacity(64),
            matched: 0,
        }
    }

    fn add_frame_data(
        &mut self,
        frame_id: u64,
        timestamp_ms: f64,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Option<Frame> {
        // Check if telemetry is already waiting
        if let Some(pending) = self.pending_telemetry.remove(&frame_id) {
            self.matched += 1;
            return Some(Frame {
                frame_id,
                timestamp_ms,
                pixels,
                width,
                height,
                telemetry: pending.telemetry,
            });
        }

        // Store frame for later matching
        self.pending_frames.insert(
            frame_id,
            PendingFrame {
                frame_id,
                timestamp_ms,
                pixels,
                width,
                height,
                added_at: Instant::now(),
            },
        );

        None
    }

    fn add_telemetry(&mut self, frame_id: u64, telemetry: Telemetry) -> Option<Frame> {
        // Check if frame is already waiting
        if let Some(pending) = self.pending_frames.remove(&frame_id) {
            self.matched += 1;
            return Some(Frame {
                frame_id: pending.frame_id,
                timestamp_ms: pending.timestamp_ms,
                pixels: pending.pixels,
                width: pending.width,
                height: pending.height,
                telemetry,
            });
        }

        // Store telemetry for later matching
        self.pending_telemetry.insert(
            frame_id,
            PendingTelemetry {
                telemetry,
                added_at: Instant::now(),
            },
        );

        None
    }
}

fn bench_frame_telemetry_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_telemetry_sync");

    // Frame arrives first (common case)
    group.bench_function("frame_then_telemetry", |b| {
        let mut sync = FrameTelemetrySync::new();
        let mut id = 0u64;

        b.iter(|| {
            let pixels = vec![0u8; 640 * 360 * 3];
            let _ = sync.add_frame_data(id, id as f64 * 200.0, pixels, 640, 360);
            let result = sync.add_telemetry(id, Telemetry::default());
            id += 1;
            black_box(result)
        });
    });

    // Telemetry arrives first
    group.bench_function("telemetry_then_frame", |b| {
        let mut sync = FrameTelemetrySync::new();
        let mut id = 0u64;

        b.iter(|| {
            let _ = sync.add_telemetry(id, Telemetry::default());
            let pixels = vec![0u8; 640 * 360 * 3];
            let result = sync.add_frame_data(id, id as f64 * 200.0, pixels, 640, 360);
            id += 1;
            black_box(result)
        });
    });

    // Out of order matching
    group.bench_function("out_of_order_10", |b| {
        b.iter(|| {
            let mut sync = FrameTelemetrySync::new();

            // Add frames 0-9
            for i in 0..10u64 {
                let pixels = vec![0u8; 640 * 360 * 3];
                sync.add_frame_data(i, i as f64 * 200.0, pixels, 640, 360);
            }

            // Add telemetry in reverse order
            let mut matched = 0;
            for i in (0..10u64).rev() {
                if sync.add_telemetry(i, Telemetry::default()).is_some() {
                    matched += 1;
                }
            }

            black_box(matched)
        });
    });

    group.finish();
}

// ============================================================================
// End-to-End Pipeline Benchmarks
// ============================================================================

/// Simulated BMP decode (just allocation + fill).
fn mock_decode(width: u32, height: u32) -> Vec<u8> {
    vec![0u8; (width as usize) * (height as usize) * 3]
}

fn bench_end_to_end_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_pipeline");

    // Full pipeline: decode -> sync -> buffer -> batch
    // Target: < 5ms
    group.bench_function("decode_sync_buffer_batch", |b| {
        let buffer = FrameBuffer::new(64);
        let (mut producer, mut consumer) = buffer.split();
        let mut sync = FrameTelemetrySync::new();
        let mut frame_id = 0u64;

        b.iter(|| {
            // Simulate decode
            let pixels = mock_decode(640, 360);

            // Add to sync
            let _ = sync.add_frame_data(
                frame_id,
                frame_id as f64 * 200.0,
                pixels,
                640,
                360,
            );

            // Add telemetry and get matched frame
            if let Some(frame) = sync.add_telemetry(frame_id, Telemetry::default()) {
                producer.push(frame).ok();
            }

            frame_id += 1;

            // Pop batch when we have enough
            if consumer.len() >= 8 {
                let batch = consumer.pop_batch(8, 640, 360);
                black_box(batch);
            }
        });
    });

    // Measure individual pipeline stages
    group.bench_function("stage_decode_only", |b| {
        b.iter(|| black_box(mock_decode(640, 360)));
    });

    group.bench_function("stage_sync_only", |b| {
        let mut sync = FrameTelemetrySync::new();
        let mut id = 0u64;

        b.iter(|| {
            let pixels = vec![0u8; 640 * 360 * 3];
            sync.add_frame_data(id, id as f64 * 200.0, pixels, 640, 360);
            let result = sync.add_telemetry(id, Telemetry::default());
            id += 1;
            black_box(result)
        });
    });

    group.bench_function("stage_buffer_only", |b| {
        let buffer = FrameBuffer::new(64);
        let (mut producer, mut consumer) = buffer.split();

        b.iter(|| {
            let frame = create_lightweight_frame(0);
            producer.push(frame).ok();
            black_box(consumer.try_pop())
        });
    });

    group.finish();
}

// ============================================================================
// Backpressure Handling Benchmarks
// ============================================================================

fn bench_backpressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("backpressure");

    // Buffer fill detection
    group.bench_function("fill_ratio_check", |b| {
        let buffer = FrameBuffer::new(256);
        let (mut producer, _consumer) = buffer.split();

        // Fill to 50%
        for i in 0..128 {
            producer.push(create_lightweight_frame(i)).unwrap();
        }

        b.iter(|| {
            black_box(producer.len() as f32 / 256.0)
        });
    });

    // Push with full buffer (retry pattern)
    group.bench_function("push_retry_pattern", |b| {
        let buffer = FrameBuffer::new(16);
        let (mut producer, mut consumer) = buffer.split();

        // Fill buffer completely
        for i in 0..16 {
            producer.push(create_lightweight_frame(i)).unwrap();
        }

        b.iter(|| {
            // Try push (will fail)
            let frame = create_lightweight_frame(100);
            if producer.push(frame).is_err() {
                // Consumer pops one
                consumer.try_pop();
            }
            // Now push succeeds
            producer.push(create_lightweight_frame(101)).ok()
        });
    });

    group.finish();
}

// ============================================================================
// Latency Distribution Benchmark
// ============================================================================

fn bench_latency_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_distribution");
    group.sample_size(1000);

    // Measure push-to-pop latency
    group.bench_function("push_to_pop_latency", |b| {
        let buffer = FrameBuffer::new(64);
        let (mut producer, mut consumer) = buffer.split();

        b.iter(|| {
            let start = Instant::now();
            let frame = create_lightweight_frame(0);
            producer.push(frame).unwrap();
            let _ = consumer.try_pop();
            black_box(start.elapsed())
        });
    });

    // Measure sync latency
    group.bench_function("sync_match_latency", |b| {
        let mut sync = FrameTelemetrySync::new();
        let mut id = 0u64;

        b.iter(|| {
            let start = Instant::now();
            let pixels = vec![0u8; 640 * 360 * 3];
            sync.add_frame_data(id, id as f64 * 200.0, pixels, 640, 360);
            sync.add_telemetry(id, Telemetry::default());
            id += 1;
            black_box(start.elapsed())
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_ring_buffer_operations,
    bench_ring_buffer_throughput,
    bench_frame_telemetry_sync,
    bench_end_to_end_pipeline,
    bench_backpressure,
    bench_latency_distribution,
);

criterion_main!(benches);
