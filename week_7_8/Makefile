NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_60

all: naive tiled demo

naive: convolution_naive.cu common.h
	$(NVCC) $(NVCC_FLAGS) -o conv_naive convolution_naive.cu
	@echo ""
	@echo "=== Running Naive Convolution ==="
	./conv_naive

tiled: convolution_tiled.cu common.h
	$(NVCC) $(NVCC_FLAGS) -o conv_tiled convolution_tiled.cu
	@echo ""
	@echo "=== Running Tiled Convolution ==="
	./conv_tiled

benchmark: convolution_naive.cu convolution_tiled.cu benchmark.cu common.h benchmark-images
	@echo "NOTE: Building benchmark requires combining kernel implementations."
	@echo "Make sure both convolution_naive.cu and convolution_tiled.cu compile."
	$(NVCC) $(NVCC_FLAGS) -o conv_benchmark benchmark.cu
	@echo ""
	@echo "=== Running Benchmark ==="
	./conv_benchmark

# Generate benchmark test images (PGM format)
benchmark-images:
	@python3 generate_sample.py --benchmark

# Visual demo with real images
demo: demo.cu convolution_tiled.cu common.h
	$(NVCC) $(NVCC_FLAGS) -o conv_demo demo.cu
	@echo ""
	@echo "=== Visual Demo ==="
	@echo "Generating sample image..."
	python3 generate_sample.py
	./conv_demo sample_input.pgm
	@echo ""
	@echo "Output files: output_blur.pgm output_gaussian.pgm output_sharpen.pgm output_edge.pgm output_emboss.pgm"
	@echo "Open them in GIMP or any image viewer to see the results."

# Run demo with your own image
demo-custom: conv_demo
	@echo "Usage: ./conv_demo <your_image.pgm> [blur|gaussian|sharpen|edge|emboss|all]"

# Generate sample image only (no GPU required)
sample:
	python3 generate_sample.py

# Test with specific sizes
test-small: conv_naive conv_tiled
	./conv_naive 256 256 3
	./conv_tiled 256 256 3

test-boundary: conv_naive conv_tiled
	./conv_naive 1000 1000 5
	./conv_tiled 1000 1000 5

clean:
	rm -f conv_naive conv_tiled conv_benchmark conv_demo
	rm -f sample_input.pgm output_*.pgm

.PHONY: all naive tiled benchmark benchmark-images demo demo-custom sample test-small test-boundary clean
