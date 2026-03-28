// Suppress dead code warnings for simulation types used in benchmarks
#![allow(dead_code)]

//! Batch loading benchmarks for DriveDiT data pipeline.
//!
//! Performance targets (from ENGINEERING_PLAN.md):
//! - Single frame load: < 1ms
//! - Parallel batch loading: > 100 fps
//! - SIMD normalization: > 1 Gpixel/sec
//!
//! Run with: cargo bench --bench batch_loading

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use rand::Rng;
use rayon::prelude::*;
use std::io::Cursor;

// ============================================================================
// Test Fixture Generation
// ============================================================================

/// Create a valid BMP file in memory.
fn create_test_bmp(width: u32, height: u32) -> Vec<u8> {
    let row_size = ((width * 3 + 3) / 4) * 4;
    let pixel_data_size = row_size * height;
    let file_size = 14 + 40 + pixel_data_size;

    let mut bmp = Vec::with_capacity(file_size as usize);

    // BMP File Header (14 bytes)
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&(file_size as u32).to_le_bytes());
    bmp.extend_from_slice(&0u16.to_le_bytes());
    bmp.extend_from_slice(&0u16.to_le_bytes());
    bmp.extend_from_slice(&54u32.to_le_bytes());

    // DIB Header (40 bytes)
    bmp.extend_from_slice(&40u32.to_le_bytes());
    bmp.extend_from_slice(&(width as i32).to_le_bytes());
    bmp.extend_from_slice(&(-(height as i32)).to_le_bytes());
    bmp.extend_from_slice(&1u16.to_le_bytes());
    bmp.extend_from_slice(&24u16.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&pixel_data_size.to_le_bytes());
    bmp.extend_from_slice(&2835u32.to_le_bytes());
    bmp.extend_from_slice(&2835u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());
    bmp.extend_from_slice(&0u32.to_le_bytes());

    // Pixel data
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 127) / (width + height).max(1)) as u8;
            bmp.push(b);
            bmp.push(g);
            bmp.push(r);
        }
        let padding = (4 - (width * 3) % 4) % 4;
        for _ in 0..padding {
            bmp.push(0);
        }
    }

    bmp
}

/// Create test frame pixels (random data).
fn create_test_frame_pixels(width: u32, height: u32) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let size = (width as usize) * (height as usize) * 3;
    let mut pixels = vec![0u8; size];
    rng.fill(&mut pixels[..]);
    pixels
}

/// Simulated frame structure.
#[derive(Clone)]
struct Frame {
    frame_id: u64,
    timestamp_ms: f64,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl Frame {
    fn new(frame_id: u64, timestamp_ms: f64, pixels: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            frame_id,
            timestamp_ms,
            pixels,
            width,
            height,
        }
    }

    fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// Simulated frame batch.
struct FrameBatch {
    pixels: Vec<u8>,
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
            frame_ids: Vec::with_capacity(capacity),
            timestamps: Vec::with_capacity(capacity),
            batch_size: 0,
            width,
            height,
        }
    }

    fn push(&mut self, frame: Frame) {
        self.pixels.extend_from_slice(&frame.pixels);
        self.frame_ids.push(frame.frame_id);
        self.timestamps.push(frame.timestamp_ms);
        self.batch_size += 1;
    }

    fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    fn clear(&mut self) {
        self.pixels.clear();
        self.frame_ids.clear();
        self.timestamps.clear();
        self.batch_size = 0;
    }
}

// ============================================================================
// Single Frame Loading Benchmarks
// ============================================================================

fn decode_bmp_bytes(data: &[u8]) -> (Vec<u8>, u32, u32) {
    let cursor = Cursor::new(data);
    let img = image::ImageReader::new(cursor)
        .with_guessed_format()
        .expect("Failed to guess format")
        .decode()
        .expect("Failed to decode");
    let rgb = img.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    let pixels = rgb.into_raw();
    (pixels, width, height)
}

fn bench_single_frame_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_frame_load");

    let bmp_data = create_test_bmp(640, 360);

    // Target: < 1ms for 640x360
    group.bench_function("decode_640x360", |b| {
        b.iter(|| decode_bmp_bytes(black_box(&bmp_data)));
    });

    // Frame construction after decode
    group.bench_function("frame_construction", |b| {
        let (pixels, width, height) = decode_bmp_bytes(&bmp_data);
        b.iter(|| {
            Frame::new(
                black_box(42),
                black_box(1234.5),
                pixels.clone(),
                width,
                height,
            )
        });
    });

    // Full pipeline: decode + construct
    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let (pixels, width, height) = decode_bmp_bytes(black_box(&bmp_data));
            Frame::new(42, 1234.5, pixels, width, height)
        });
    });

    group.finish();
}

// ============================================================================
// Parallel Batch Loading Benchmarks
// ============================================================================

fn bench_parallel_batch_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_batch_loading");

    // Create multiple BMP files in memory
    let bmp_data = create_test_bmp(640, 360);
    let batch_sizes = [4, 8, 16, 32];

    for batch_size in batch_sizes {
        let bmp_batch: Vec<Vec<u8>> = (0..batch_size).map(|_| bmp_data.clone()).collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        // Sequential loading
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &bmp_batch,
            |b, batch| {
                b.iter(|| {
                    let mut frames = Vec::with_capacity(batch.len());
                    for (i, data) in batch.iter().enumerate() {
                        let (pixels, width, height) = decode_bmp_bytes(black_box(data));
                        frames.push(Frame::new(i as u64, i as f64 * 200.0, pixels, width, height));
                    }
                    black_box(frames)
                });
            },
        );

        // Parallel loading with rayon
        group.bench_with_input(
            BenchmarkId::new("parallel_rayon", batch_size),
            &bmp_batch,
            |b, batch| {
                b.iter(|| {
                    let frames: Vec<Frame> = batch
                        .par_iter()
                        .enumerate()
                        .map(|(i, data)| {
                            let (pixels, width, height) = decode_bmp_bytes(black_box(data));
                            Frame::new(i as u64, i as f64 * 200.0, pixels, width, height)
                        })
                        .collect();
                    black_box(frames)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SIMD Normalization Benchmarks
// ============================================================================

/// Scalar u8 to f32 normalization.
fn normalize_scalar(input: &[u8], output: &mut [f32]) {
    for (i, &val) in input.iter().enumerate() {
        output[i] = val as f32 / 255.0;
    }
}

/// SIMD-friendly u8 to f32 normalization using chunks.
fn normalize_chunked(input: &[u8], output: &mut [f32]) {
    const CHUNK_SIZE: usize = 8;
    let chunks = input.len() / CHUNK_SIZE;

    for chunk_idx in 0..chunks {
        let start = chunk_idx * CHUNK_SIZE;
        for i in 0..CHUNK_SIZE {
            output[start + i] = input[start + i] as f32 / 255.0;
        }
    }

    // Handle remainder
    let remainder_start = chunks * CHUNK_SIZE;
    for i in remainder_start..input.len() {
        output[i] = input[i] as f32 / 255.0;
    }
}

/// Use bytemuck for more efficient conversion (if applicable).
fn normalize_bytemuck(input: &[u8], output: &mut [f32]) {
    // This is essentially the same as scalar but demonstrates the pattern
    input
        .iter()
        .zip(output.iter_mut())
        .for_each(|(&src, dst)| {
            *dst = src as f32 * (1.0 / 255.0);
        });
}

fn bench_simd_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_normalization");

    let sizes = [
        (320 * 180 * 3, "320x180"),
        (640 * 360 * 3, "640x360"),
        (1280 * 720 * 3, "1280x720"),
    ];

    for (size, name) in sizes {
        let input = create_test_frame_pixels(640, 360);

        group.throughput(Throughput::Elements(size as u64));

        // Scalar normalization
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &input,
            |b, input| {
                let mut output = vec![0.0f32; input.len()];
                b.iter(|| {
                    normalize_scalar(black_box(input), &mut output);
                    black_box(output.len())
                });
            },
        );

        // Chunked normalization (auto-vectorizable)
        group.bench_with_input(
            BenchmarkId::new("chunked", name),
            &input,
            |b, input| {
                let mut output = vec![0.0f32; input.len()];
                b.iter(|| {
                    normalize_chunked(black_box(input), &mut output);
                    black_box(output.len())
                });
            },
        );

        // Bytemuck style
        group.bench_with_input(
            BenchmarkId::new("bytemuck", name),
            &input,
            |b, input| {
                let mut output = vec![0.0f32; input.len()];
                b.iter(|| {
                    normalize_bytemuck(black_box(input), &mut output);
                    black_box(output.len())
                });
            },
        );

        // Parallel normalization with rayon
        group.bench_with_input(
            BenchmarkId::new("parallel", name),
            &input,
            |b, input| {
                let mut output = vec![0.0f32; input.len()];
                b.iter(|| {
                    input
                        .par_iter()
                        .zip(output.par_iter_mut())
                        .for_each(|(&src, dst)| {
                            *dst = src as f32 * (1.0 / 255.0);
                        });
                    black_box(output.len())
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Session Iteration Benchmarks
// ============================================================================

/// Simulated session with pre-loaded frames.
struct Session {
    frames: Vec<Frame>,
    current_index: usize,
}

impl Session {
    fn new(num_frames: usize) -> Self {
        let frames: Vec<Frame> = (0..num_frames)
            .map(|i| {
                Frame::new(
                    i as u64,
                    i as f64 * 200.0,
                    create_test_frame_pixels(640, 360),
                    640,
                    360,
                )
            })
            .collect();

        Self {
            frames,
            current_index: 0,
        }
    }

    fn next_frame(&mut self) -> Option<&Frame> {
        if self.current_index < self.frames.len() {
            let frame = &self.frames[self.current_index];
            self.current_index += 1;
            Some(frame)
        } else {
            None
        }
    }

    fn get_batch(&mut self, batch_size: usize) -> Option<FrameBatch> {
        if self.current_index + batch_size > self.frames.len() {
            return None;
        }

        let mut batch = FrameBatch::with_capacity(batch_size, 640, 360);
        for _ in 0..batch_size {
            if let Some(frame) = self.next_frame() {
                batch.push(frame.clone());
            }
        }

        Some(batch)
    }

    fn reset(&mut self) {
        self.current_index = 0;
    }
}

fn bench_session_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("session_iteration");

    let session_sizes = [100, 500, 1000];
    let batch_sizes = [4, 8, 16];

    for session_size in session_sizes {
        // Single frame iteration
        group.throughput(Throughput::Elements(session_size as u64));
        group.bench_with_input(
            BenchmarkId::new("single_frame", session_size),
            &session_size,
            |b, &size| {
                let mut session = Session::new(size);
                b.iter(|| {
                    session.reset();
                    let mut count = 0;
                    while let Some(frame) = session.next_frame() {
                        black_box(frame);
                        count += 1;
                    }
                    count
                });
            },
        );

        // Batch iteration
        for batch_size in batch_sizes {
            let batches_per_session = session_size / batch_size;
            group.throughput(Throughput::Elements(batches_per_session as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("batch_{}", batch_size), session_size),
                &session_size,
                |b, &size| {
                    let mut session = Session::new(size);
                    b.iter(|| {
                        session.reset();
                        let mut count = 0;
                        while let Some(batch) = session.get_batch(batch_size) {
                            black_box(&batch);
                            count += 1;
                        }
                        count
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Batch Assembly Benchmarks
// ============================================================================

fn bench_batch_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_assembly");

    let batch_sizes = [4, 8, 16, 32];

    for batch_size in batch_sizes {
        // Pre-create frames
        let frames: Vec<Frame> = (0..batch_size)
            .map(|i| {
                Frame::new(
                    i as u64,
                    i as f64 * 200.0,
                    create_test_frame_pixels(640, 360),
                    640,
                    360,
                )
            })
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));

        // Push frames to batch one by one
        group.bench_with_input(
            BenchmarkId::new("sequential_push", batch_size),
            &frames,
            |b, frames| {
                b.iter(|| {
                    let mut batch = FrameBatch::with_capacity(frames.len(), 640, 360);
                    for frame in frames.iter().cloned() {
                        batch.push(frame);
                    }
                    black_box(batch)
                });
            },
        );

        // Pre-allocate and bulk extend
        group.bench_with_input(
            BenchmarkId::new("bulk_extend", batch_size),
            &frames,
            |b, frames| {
                b.iter(|| {
                    let frame_size = 640 * 360 * 3;
                    let mut pixels = Vec::with_capacity(frames.len() * frame_size);
                    let mut frame_ids = Vec::with_capacity(frames.len());
                    let mut timestamps = Vec::with_capacity(frames.len());

                    for frame in frames.iter() {
                        pixels.extend_from_slice(&frame.pixels);
                        frame_ids.push(frame.frame_id);
                        timestamps.push(frame.timestamp_ms);
                    }

                    black_box((pixels, frame_ids, timestamps))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Memory Layout Benchmarks
// ============================================================================

fn bench_memory_layout(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout");

    // Compare HWC vs CHW memory access patterns
    let width: usize = 640;
    let height: usize = 360;
    let channels: usize = 3;
    let total_size = width * height * channels;

    let hwc_data: Vec<u8> = create_test_frame_pixels(width as u32, height as u32);

    // HWC to CHW conversion
    group.throughput(Throughput::Elements(total_size as u64));

    group.bench_function("hwc_to_chw", |b| {
        let mut chw_data = vec![0u8; total_size];
        b.iter(|| {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let hwc_idx = h * width * channels + w * channels + c;
                        let chw_idx = c * height * width + h * width + w;
                        chw_data[chw_idx] = hwc_data[hwc_idx];
                    }
                }
            }
            black_box(chw_data.len())
        });
    });

    // Parallel HWC to CHW
    group.bench_function("hwc_to_chw_parallel", |b| {
        let mut chw_data = vec![0u8; total_size];
        b.iter(|| {
            chw_data
                .par_chunks_mut(height * width)
                .enumerate()
                .for_each(|(c, channel_slice)| {
                    for h in 0..height {
                        for w in 0..width {
                            let hwc_idx = h * width * channels + w * channels + c;
                            let chw_idx = h * width + w;
                            channel_slice[chw_idx] = hwc_data[hwc_idx];
                        }
                    }
                });
            black_box(chw_data.len())
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_single_frame_load,
    bench_parallel_batch_loading,
    bench_simd_normalization,
    bench_session_iteration,
    bench_batch_assembly,
    bench_memory_layout,
);

criterion_main!(benches);
