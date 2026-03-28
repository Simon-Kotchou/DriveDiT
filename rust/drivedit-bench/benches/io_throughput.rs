//! I/O throughput benchmarks for DriveDiT data pipeline.
//!
//! Performance targets (from ENGINEERING_PLAN.md):
//! - Frame decode latency: < 1ms (BMP 640x360)
//! - JPEG decode: < 2ms (640x360)
//! - CSV parse throughput: > 100k rows/sec
//!
//! Run with: cargo bench --bench io_throughput

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use image::{ImageBuffer, ImageFormat, Rgb};
use std::io::Cursor;

// ============================================================================
// Test Fixture Generation
// ============================================================================

/// Create a valid BMP file in memory.
///
/// This generates a minimal but valid BMP file with the specified dimensions.
/// The pixel data is filled with a gradient pattern.
fn create_test_bmp(width: u32, height: u32) -> Vec<u8> {
    // BMP file format:
    // - File header: 14 bytes
    // - DIB header (BITMAPINFOHEADER): 40 bytes
    // - Pixel data: width * height * 3 bytes (24-bit RGB), padded to 4-byte rows

    let row_size = ((width * 3 + 3) / 4) * 4; // Row size padded to 4 bytes
    let pixel_data_size = row_size * height;
    let file_size = 14 + 40 + pixel_data_size;

    let mut bmp = Vec::with_capacity(file_size as usize);

    // BMP File Header (14 bytes)
    bmp.extend_from_slice(b"BM"); // Magic
    bmp.extend_from_slice(&(file_size as u32).to_le_bytes()); // File size
    bmp.extend_from_slice(&0u16.to_le_bytes()); // Reserved
    bmp.extend_from_slice(&0u16.to_le_bytes()); // Reserved
    bmp.extend_from_slice(&54u32.to_le_bytes()); // Pixel data offset (14 + 40)

    // DIB Header - BITMAPINFOHEADER (40 bytes)
    bmp.extend_from_slice(&40u32.to_le_bytes()); // Header size
    bmp.extend_from_slice(&(width as i32).to_le_bytes()); // Width
    bmp.extend_from_slice(&(-(height as i32)).to_le_bytes()); // Height (negative = top-down)
    bmp.extend_from_slice(&1u16.to_le_bytes()); // Color planes
    bmp.extend_from_slice(&24u16.to_le_bytes()); // Bits per pixel
    bmp.extend_from_slice(&0u32.to_le_bytes()); // Compression (none)
    bmp.extend_from_slice(&pixel_data_size.to_le_bytes()); // Image size
    bmp.extend_from_slice(&2835u32.to_le_bytes()); // Horizontal resolution (72 DPI)
    bmp.extend_from_slice(&2835u32.to_le_bytes()); // Vertical resolution
    bmp.extend_from_slice(&0u32.to_le_bytes()); // Colors in palette
    bmp.extend_from_slice(&0u32.to_le_bytes()); // Important colors

    // Pixel data (BGR format, top-down with negative height)
    for y in 0..height {
        for x in 0..width {
            // Create a simple gradient pattern
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 127) / (width + height).max(1)) as u8;

            // BMP uses BGR order
            bmp.push(b);
            bmp.push(g);
            bmp.push(r);
        }
        // Pad row to 4-byte boundary
        let padding = (4 - (width * 3) % 4) % 4;
        for _ in 0..padding {
            bmp.push(0);
        }
    }

    bmp
}

/// Create a JPEG file in memory using the image crate.
fn create_test_jpeg(width: u32, height: u32, quality: u8) -> Vec<u8> {
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    // Fill with gradient pattern
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = ((x * 255) / width.max(1)) as u8;
        let g = ((y * 255) / height.max(1)) as u8;
        let b = (((x + y) * 127) / (width + height).max(1)) as u8;
        *pixel = Rgb([r, g, b]);
    }

    let mut buffer = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, quality);
    encoder
        .encode_image(&img)
        .expect("Failed to encode JPEG");

    buffer
}

/// Create a PNG file in memory using the image crate.
fn create_test_png(width: u32, height: u32) -> Vec<u8> {
    let mut img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    // Fill with gradient pattern
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = ((x * 255) / width.max(1)) as u8;
        let g = ((y * 255) / height.max(1)) as u8;
        let b = (((x + y) * 127) / (width + height).max(1)) as u8;
        *pixel = Rgb([r, g, b]);
    }

    let mut buffer = Vec::new();
    img.write_to(&mut Cursor::new(&mut buffer), ImageFormat::Png)
        .expect("Failed to encode PNG");

    buffer
}

/// Create test telemetry CSV content.
fn create_test_telemetry_csv(num_frames: usize) -> String {
    let mut csv = String::from(
        "FrameID,Timestamp_ms,Speed_kmh,Steering,Throttle,Brake,\
         PosX,PosY,PosZ,ForwardX,ForwardY,ForwardZ,UpX,UpY,UpZ,RightX,RightY,RightZ,\
         Acceleration,DistanceTotal\n",
    );

    for i in 0..num_frames {
        let frame_id = i as u64;
        let timestamp = i as f64 * 200.0;
        let pos_x = 100.0 + i as f32 * 0.5;
        let speed = 60.0 + (i as f32 * 0.1).sin() * 10.0;
        let steering = (i as f32 * 0.05).sin() * 0.3;

        csv.push_str(&format!(
            "{},{},{},{},0.8,0.0,{},0.0,200.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.5,{}\n",
            frame_id,
            timestamp,
            speed,
            steering,
            pos_x,
            i as f32 * 3.33
        ));
    }

    csv
}

// ============================================================================
// BMP Decode Benchmarks
// ============================================================================

/// Decode BMP from bytes using image crate (reference implementation).
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

fn bench_bmp_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bmp_decode");

    // Test various resolutions
    let resolutions = [
        (320, 180, "320x180"),
        (640, 360, "640x360_default"),
        (1280, 720, "1280x720_hd"),
        (1920, 1080, "1920x1080_fhd"),
    ];

    for (width, height, name) in resolutions {
        let bmp_data = create_test_bmp(width, height);
        let pixels = (width * height) as u64;
        let bytes = bmp_data.len() as u64;

        // Throughput by pixels
        group.throughput(Throughput::Elements(pixels));
        group.bench_with_input(BenchmarkId::new("pixels", name), &bmp_data, |b, data| {
            b.iter(|| decode_bmp_bytes(black_box(data)));
        });

        // Throughput by bytes
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::new("bytes", name), &bmp_data, |b, data| {
            b.iter(|| decode_bmp_bytes(black_box(data)));
        });
    }

    group.finish();
}

/// Dedicated benchmark for the target resolution (640x360).
/// Target: < 1ms
fn bench_bmp_decode_target(c: &mut Criterion) {
    let bmp_data = create_test_bmp(640, 360);

    c.bench_function("bmp_decode_640x360_target_1ms", |b| {
        b.iter(|| decode_bmp_bytes(black_box(&bmp_data)));
    });
}

// ============================================================================
// JPEG Decode Benchmarks
// ============================================================================

fn bench_jpeg_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("jpeg_decode");

    // Test various resolutions and quality levels
    let resolutions = [(640, 360), (1280, 720), (1920, 1080)];
    let qualities = [75, 90, 100];

    for (width, height) in resolutions {
        for quality in qualities {
            let jpeg_data = create_test_jpeg(width, height, quality);
            let pixels = (width * height) as u64;
            let bytes = jpeg_data.len() as u64;
            let name = format!("{}x{}_q{}", width, height, quality);

            group.throughput(Throughput::Elements(pixels));
            group.bench_with_input(
                BenchmarkId::new("decode", &name),
                &jpeg_data,
                |b, data| {
                    b.iter(|| {
                        let cursor = Cursor::new(black_box(data));
                        let img = image::ImageReader::new(cursor)
                            .with_guessed_format()
                            .expect("Failed to guess format")
                            .decode()
                            .expect("Failed to decode");
                        let rgb = img.to_rgb8();
                        black_box(rgb.into_raw())
                    });
                },
            );

            // Also measure compression ratio
            let compression_ratio = (width * height * 3) as f64 / bytes as f64;
            eprintln!(
                "JPEG {} compression ratio: {:.2}x",
                name, compression_ratio
            );
        }
    }

    group.finish();
}

// ============================================================================
// PNG Decode Benchmarks
// ============================================================================

fn bench_png_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("png_decode");

    let resolutions = [(640, 360), (1280, 720)];

    for (width, height) in resolutions {
        let png_data = create_test_png(width, height);
        let pixels = (width * height) as u64;
        let name = format!("{}x{}", width, height);

        group.throughput(Throughput::Elements(pixels));
        group.bench_with_input(BenchmarkId::new("decode", &name), &png_data, |b, data| {
            b.iter(|| {
                let cursor = Cursor::new(black_box(data));
                let img = image::ImageReader::new(cursor)
                    .with_guessed_format()
                    .expect("Failed to guess format")
                    .decode()
                    .expect("Failed to decode");
                let rgb = img.to_rgb8();
                black_box(rgb.into_raw())
            });
        });
    }

    group.finish();
}

// ============================================================================
// CSV Parse Benchmarks
// ============================================================================

fn bench_csv_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_parse");

    // Test different CSV sizes
    let sizes = [100, 1000, 10000, 100000];

    for size in sizes {
        let csv_content = create_test_telemetry_csv(size);
        let bytes = csv_content.len() as u64;

        // Throughput by rows
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("rows", size),
            &csv_content,
            |b, csv| {
                b.iter(|| {
                    let mut count = 0;
                    for line in csv.lines().skip(1) {
                        // Skip header
                        let fields: Vec<&str> = line.split(',').collect();
                        // Parse key fields
                        let _frame_id: u64 = fields[0].parse().unwrap_or(0);
                        let _timestamp: f64 = fields[1].parse().unwrap_or(0.0);
                        let _speed: f32 = fields[2].parse().unwrap_or(0.0);
                        let _steering: f32 = fields[3].parse().unwrap_or(0.0);
                        count += 1;
                    }
                    black_box(count)
                });
            },
        );

        // Throughput by bytes
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::new("bytes", size),
            &csv_content,
            |b, csv| {
                b.iter(|| {
                    let mut count = 0;
                    for line in csv.lines().skip(1) {
                        let fields: Vec<&str> = line.split(',').collect();
                        let _frame_id: u64 = fields[0].parse().unwrap_or(0);
                        count += 1;
                    }
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CSV parse with full struct construction.
fn bench_csv_parse_full(c: &mut Criterion) {
    let csv_content = create_test_telemetry_csv(10000);

    c.bench_function("csv_parse_full_10k", |b| {
        b.iter(|| {
            let mut telemetry_records = Vec::with_capacity(10000);
            for line in csv_content.lines().skip(1) {
                let fields: Vec<&str> = line.split(',').collect();
                if fields.len() >= 20 {
                    let record = (
                        fields[0].parse::<u64>().unwrap_or(0),    // frame_id
                        fields[1].parse::<f64>().unwrap_or(0.0),  // timestamp
                        fields[2].parse::<f32>().unwrap_or(0.0),  // speed
                        fields[3].parse::<f32>().unwrap_or(0.0),  // steering
                        fields[4].parse::<f32>().unwrap_or(0.0),  // throttle
                        fields[5].parse::<f32>().unwrap_or(0.0),  // brake
                        fields[6].parse::<f32>().unwrap_or(0.0),  // pos_x
                        fields[7].parse::<f32>().unwrap_or(0.0),  // pos_y
                        fields[8].parse::<f32>().unwrap_or(0.0),  // pos_z
                    );
                    telemetry_records.push(record);
                }
            }
            black_box(telemetry_records)
        });
    });
}

// ============================================================================
// Memory Allocation Benchmarks
// ============================================================================

fn bench_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation");

    // Frame buffer allocation
    let frame_sizes = [
        (320, 180, "320x180"),
        (640, 360, "640x360"),
        (1280, 720, "1280x720"),
        (1920, 1080, "1920x1080"),
    ];

    for (width, height, name) in frame_sizes {
        let size = (width as usize) * (height as usize) * 3;

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("frame_buffer", name),
            &size,
            |b, &size| {
                b.iter(|| {
                    let buffer = vec![0u8; size];
                    black_box(buffer)
                });
            },
        );
    }

    // Pre-allocated vs fresh allocation
    group.bench_function("preallocated_640x360", |b| {
        let mut buffer = vec![0u8; 640 * 360 * 3];
        b.iter(|| {
            // Just fill existing buffer
            buffer.fill(0);
            black_box(buffer.len())
        });
    });

    group.finish();
}

// ============================================================================
// BMP Header Parse Benchmark
// ============================================================================

/// Validate that bytes represent a valid BMP.
fn is_valid_bmp(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == b'B' && data[1] == b'M'
}

/// Extract BMP dimensions from header without full decode.
fn bmp_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 26 || !is_valid_bmp(data) {
        return None;
    }

    let width = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let height = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);

    Some((width as u32, height.unsigned_abs()))
}

fn bench_bmp_header_parse(c: &mut Criterion) {
    let bmp_data = create_test_bmp(640, 360);

    let mut group = c.benchmark_group("bmp_header");

    group.bench_function("is_valid", |b| {
        b.iter(|| is_valid_bmp(black_box(&bmp_data)));
    });

    group.bench_function("dimensions", |b| {
        b.iter(|| bmp_dimensions(black_box(&bmp_data)));
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    benches,
    bench_bmp_decode,
    bench_bmp_decode_target,
    bench_jpeg_decode,
    bench_png_decode,
    bench_csv_parse,
    bench_csv_parse_full,
    bench_allocation,
    bench_bmp_header_parse,
);

criterion_main!(benches);
