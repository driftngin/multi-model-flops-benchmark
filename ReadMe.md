# Multi-Model FLOPS Benchmark Tool

A simple GUI tool to benchmark your hardware's FLOPS (Floating Point Operations Per Second) performance using popular neural network models.

<img width="800" height="630" alt="FLOPS-benchmark" src="https://github.com/user-attachments/assets/d0a9b135-b467-40b0-9092-32963e1aef61" />


## Features

- **Simple Mode**: Quick ResNet50 benchmark (offline, no downloads)
- **Advanced Mode**: Test multiple models (ResNet50, EfficientNet-B0, ViT-B/16)
- **Real-time progress tracking** and detailed performance metrics
- **Cross-platform** GUI using tkinter (built into Python)
- **Offline capable** - works without internet connection

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python benchmark_gui.py
   ```

3. **Click "Run Benchmark"** for instant results, or enable "Advanced Mode" for more options.

## Requirements

- Python 3.8 or higher
- ~2GB RAM for model loading
- Works with CPU and compatible GPUs

## Performance Metrics

The tool provides:
- **Runtime TFLOPS** - Actual throughput on your hardware
- **Hardware utilization** - Efficiency percentage
- **Throughput** - Samples processed per second
- **Timing** - Per-sample and per-batch measurements

## Models Tested

- **ResNet50**: Traditional CNN (available offline)
- **EfficientNet-B0**: Modern efficient architecture 
- **ViT-B/16**: Vision Transformer (attention-based)

## Notes

- Simple mode uses ResNet50 with random weights (no downloads required)
- Advanced mode can download pretrained weights for more comprehensive testing
- Results can be saved as text files or CSV for further analysis

---

**Perfect for:** Hardware evaluation, GPU benchmarking, AI workload testing
