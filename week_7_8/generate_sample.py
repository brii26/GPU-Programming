#!/usr/bin/env python3
"""
Generate a sample grayscale test image (PGM format) for the convolution demo.

Creates a 512x512 image with geometric shapes, gradients, and textures
so that different convolution filters produce visually distinct results:
  - Box blur    → smooths everything uniformly
  - Gaussian    → smooths with softer falloff
  - Sharpen     → makes edges crisper
  - Edge detect → highlights boundaries between regions
  - Emboss      → gives a 3D relief appearance

Usage: python3 generate_sample.py [output_file] [size]
  Default: sample_input.pgm, 512x512

No external dependencies required (uses only math + struct).
"""

import math
import sys
import struct

def generate_sample_image(width=512, height=512):
    """Generate a test image with geometric shapes and gradients."""
    pixels = [0] * (width * height)

    for y in range(height):
        for x in range(width):
            val = 0.0

            # Background: subtle diagonal gradient
            val = 0.15 + 0.1 * (x + y) / (width + height)

            # Large circle (bright) - top left area
            cx, cy, r = width * 0.3, height * 0.35, min(width, height) * 0.18
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < r:
                val = 0.85 - 0.3 * (dist / r)  # radial gradient inside circle

            # Rectangle with sharp edges - right side
            rx1, ry1 = int(width * 0.55), int(height * 0.15)
            rx2, ry2 = int(width * 0.85), int(height * 0.50)
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                val = 0.7

            # Smaller bright square inside the rectangle
            sx1, sy1 = int(width * 0.63), int(height * 0.22)
            sx2, sy2 = int(width * 0.77), int(height * 0.42)
            if sx1 <= x <= sx2 and sy1 <= y <= sy2:
                val = 0.95

            # Triangle - bottom left
            tx, ty = width * 0.25, height * 0.85  # bottom vertex
            tsize = min(width, height) * 0.2
            # Check if point is inside triangle
            ty_top = ty - tsize
            if ty_top <= y <= ty:
                half_width_at_y = tsize * 0.6 * (y - ty_top) / tsize
                if abs(x - tx) <= half_width_at_y:
                    val = 0.6

            # Horizontal stripes region - bottom right
            if x > width * 0.5 and y > height * 0.55:
                stripe = int((y - height * 0.55) / (height * 0.04))
                if stripe % 2 == 0:
                    val = 0.8
                else:
                    val = 0.3

            # Checkerboard region - bottom center
            if width * 0.25 <= x <= width * 0.5 and y > height * 0.6:
                bx = int((x - width * 0.25) / (width * 0.03))
                by = int((y - height * 0.6) / (height * 0.03))
                if (bx + by) % 2 == 0:
                    val = 0.9
                else:
                    val = 0.1

            # Small bright dots (to test sharpen/edge response)
            dots = [
                (width * 0.1, height * 0.1),
                (width * 0.15, height * 0.12),
                (width * 0.9, height * 0.9),
                (width * 0.85, height * 0.85),
                (width * 0.5, height * 0.5),
            ]
            for dx, dy in dots:
                if abs(x - dx) <= 2 and abs(y - dy) <= 2:
                    val = 1.0

            # Concentric rings - center area
            ring_cx, ring_cy = width * 0.5, height * 0.5
            ring_dist = math.sqrt((x - ring_cx)**2 + (y - ring_cy)**2)
            ring_zone = min(width, height) * 0.12
            if ring_zone * 0.3 < ring_dist < ring_zone:
                ring_val = math.sin(ring_dist / (ring_zone * 0.08) * math.pi)
                if ring_val > 0:
                    val = max(val, 0.4 + 0.4 * ring_val)

            # Clamp and store
            val = max(0.0, min(1.0, val))
            pixels[y * width + x] = int(val * 255)

    return pixels

def write_pgm(filename, pixels, width, height):
    """Write a P5 (binary) PGM file."""
    with open(filename, 'wb') as f:
        header = f"P5\n# Generated test image for convolution demo\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))
        f.write(struct.pack(f'{width * height}B', *pixels))
    print(f"Generated: {filename} ({width}x{height})")

def main():
    # Check for --benchmark flag to generate all benchmark images
    if len(sys.argv) >= 2 and sys.argv[1] == "--benchmark":
        generate_benchmark_images()
        return

    # Single image mode
    output = "sample_input.pgm"
    size = 512

    if len(sys.argv) >= 2:
        output = sys.argv[1]
    if len(sys.argv) >= 3:
        size = int(sys.argv[2])

    print(f"Generating {size}x{size} sample image...")
    pixels = generate_sample_image(size, size)
    write_pgm(output, pixels, size, size)

    print(f"\nFeatures in the image:")
    print(f"  - Diagonal gradient background  (tests blur smoothing)")
    print(f"  - Large circle with radial fade  (tests edge detection)")
    print(f"  - Sharp-edged rectangles          (tests sharpen/edge)")
    print(f"  - Triangle                        (tests diagonal edges)")
    print(f"  - Horizontal stripes              (tests directional blur)")
    print(f"  - Checkerboard pattern            (tests high-frequency response)")
    print(f"  - Concentric rings                (tests circular edge detection)")
    print(f"  - Small bright dots               (tests point response)")
    print(f"\nRun the demo:  ./conv_demo {output}")


def generate_benchmark_images():
    """Generate all PGM images needed for the benchmark configurations."""
    import os
    os.makedirs("images", exist_ok=True)

    # These match the BenchConfig entries in benchmark.cu
    sizes = [
        (512,  512,  "benchmark_512x512"),
        (2048, 2048, "benchmark_2048x2048"),
        (4096, 4096, "benchmark_4096x4096"),
        (1000, 1000, "benchmark_1000x1000"),
    ]

    print("Generating benchmark test images...")
    print("(This may take a minute for the 4096x4096 image)\n")

    for w, h, name in sizes:
        filepath = f"images/{name}.pgm"
        if os.path.exists(filepath):
            print(f"  Already exists: {filepath} ({w}x{h}) - skipping")
            continue
        print(f"  Generating {w}x{h}...", end=" ", flush=True)
        pixels = generate_sample_image(w, h)
        write_pgm(filepath, pixels, w, h)

    # Also generate the demo sample if it doesn't exist
    if not os.path.exists("sample_input.pgm"):
        print(f"  Generating demo image...", end=" ", flush=True)
        pixels = generate_sample_image(512, 512)
        write_pgm("sample_input.pgm", pixels, 512, 512)

    print(f"\nAll benchmark images ready in images/ directory.")
    print(f"Run the benchmark:  make benchmark")


if __name__ == "__main__":
    main()
