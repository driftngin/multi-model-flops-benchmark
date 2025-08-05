import time
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import ssl
from ptflops import get_model_complexity_info
from torchvision.models import resnet50, efficientnet_b0, vit_b_16

# Handle SSL certificate issues for model downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class ModelBenchmarkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Model FLOPS Benchmark Tool - by DriftNgin")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        # Available models
        self.models = {
            "ResNet50": {
                "model_fn": lambda: self._load_model_safe(resnet50),
                "input_size": (3, 224, 224),
                "description": "Deep residual network - conv2d heavy"
            },
            "EfficientNet-B0": {
                "model_fn": lambda: self._load_model_safe(efficientnet_b0),
                "input_size": (3, 224, 224),
                "description": "Efficient architecture - mixed operations"
            },
            "ViT-B/16": {
                "model_fn": lambda: self._load_model_safe(vit_b_16),
                "input_size": (3, 224, 224),
                "description": "Vision Transformer - attention heavy"
            }
        }
        
        # Variables
        self.is_running = False
        self.benchmark_thread = None
        self.results_data = {}
        self.advanced_mode = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
        # Initialize UI state
        self.toggle_advanced_mode()
        
    def toggle_advanced_mode(self):
        """Toggle between simple and advanced mode"""
        if self.advanced_mode.get():
            # Show advanced controls
            self.model_frame.grid()
            self.settings_frame.grid()
            self.simple_desc_frame.grid_remove()
        else:
            # Hide advanced controls and show simple description
            self.model_frame.grid_remove()
            self.settings_frame.grid_remove()
            self.simple_desc_frame.grid()
            
            # Reset to default values in simple mode
            self.runs_var.set("50")  # Fewer runs for quicker results
            self.warmup_var.set("5")
            self.batch_var.set("1")
            self.precision_var.set("FP32")
            self.offline_var.set(True)  # Default to offline for simplicity
            
            # Select all models in simple mode
            for var in self.model_vars.values():
                var.set(True)
                
        # Update button visibility
        self.update_button_visibility()
                
    def update_button_visibility(self):
        """Show/hide buttons based on mode"""
        if self.advanced_mode.get():
            # Show all buttons in advanced mode
            self.save_button.pack(side=tk.LEFT, padx=(0, 10))
            self.test_button.pack(side=tk.LEFT, padx=(0, 10))
            self.export_button.pack(side=tk.LEFT)
        else:
            # Hide advanced buttons in simple mode
            self.save_button.pack_forget()
            self.test_button.pack_forget()
            self.export_button.pack_forget()
    
    def test_models_on_startup(self):
        """Test if models can be loaded successfully"""
        def test_thread():
            try:
                self.log_output("🔧 Initializing Multi-Model FLOPS Benchmark Tool...")
                self.log_output("Testing model availability...")
                
                # Test each model quickly
                success_count = 0
                for model_name, model_info in self.models.items():
                    try:
                        model = model_info["model_fn"]()
                        del model  # Free memory
                        self.log_output(f"✅ {model_name} - Ready")
                        success_count += 1
                    except Exception as e:
                        self.log_output(f"❌ {model_name} - Error: {str(e)[:50]}...")
                        
                if success_count == len(self.models):
                    self.log_output("All models ready! Click 'Run Benchmark' to start.")
                else:
                    self.log_output("⚠️  Some models failed. Try enabling 'Advanced Mode' → 'Offline mode'")
                    
                self.log_output("💡 Tip: Use 'Advanced Mode' for custom settings and detailed options.")
                
            except Exception as e:
                self.log_output(f"❌ Startup test error: {str(e)}")
                
        # Run test in background
        threading.Thread(target=test_thread, daemon=True).start()
    
    def _load_model_safe(self, model_fn):
        """Safely load model with fallback options"""
        if self.offline_var.get():
            # Force offline mode - no pretrained weights
            try:
                return model_fn(weights=None)
            except:
                try:
                    return model_fn(pretrained=False)
                except:
                    return model_fn()
        
        try:
            # Try loading with pretrained weights first
            return model_fn(weights='DEFAULT')
        except Exception as e1:
            self.log_output(f"Warning: Failed to download pretrained weights, trying fallback...")
            try:
                # Fallback to pretrained=True (older API)
                return model_fn(pretrained=True)
            except Exception as e2:
                try:
                    # Fallback to no pretrained weights
                    self.log_output(f"Warning: Loading model without pretrained weights")
                    return model_fn(weights=None)
                except Exception as e3:
                    try:
                        # Final fallback with old API
                        return model_fn(pretrained=False)
                    except Exception as e4:
                        # Last resort - basic instantiation
                        return model_fn()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)
        
        # Title and mode toggle
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="Multi-Model FLOPS Benchmark Tool", 
                               font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Advanced mode toggle
        advanced_toggle = ttk.Checkbutton(title_frame, text="Advanced Mode", 
                                         variable=self.advanced_mode,
                                         command=self.toggle_advanced_mode)
        advanced_toggle.pack(side=tk.RIGHT, padx=(20, 0))
        
        # Model selection frame (Advanced mode only)
        self.model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="10")
        self.model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        self.model_frame.columnconfigure(1, weight=1)
        
        # Model checkboxes
        self.model_vars = {}
        row = 0
        for model_name, model_info in self.models.items():
            var = tk.BooleanVar(value=True)  # All selected by default
            self.model_vars[model_name] = var
            
            checkbox = ttk.Checkbutton(self.model_frame, text=model_name, variable=var)
            checkbox.grid(row=row, column=0, sticky=tk.W, padx=(0, 20))
            
            desc_label = ttk.Label(self.model_frame, text=model_info["description"], 
                                  foreground="gray")
            desc_label.grid(row=row, column=1, sticky=tk.W)
            row += 1
        
        # Settings frame (Advanced mode only)
        self.settings_frame = ttk.LabelFrame(main_frame, text="Benchmark Settings", padding="10")
        self.settings_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        self.settings_frame.columnconfigure(1, weight=1)
        
        # Number of runs
        ttk.Label(self.settings_frame, text="Number of runs:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.runs_var = tk.StringVar(value="100")
        runs_spinbox = ttk.Spinbox(self.settings_frame, from_=10, to=1000, width=10, 
                                  textvariable=self.runs_var)
        runs_spinbox.grid(row=0, column=1, sticky=tk.W)
        
        # Warmup runs
        ttk.Label(self.settings_frame, text="Warmup runs:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.warmup_var = tk.StringVar(value="10")
        warmup_spinbox = ttk.Spinbox(self.settings_frame, from_=0, to=50, width=10,
                                    textvariable=self.warmup_var)
        warmup_spinbox.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
        
        # Batch size
        ttk.Label(self.settings_frame, text="Batch size:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.batch_var = tk.StringVar(value="1")
        batch_combo = ttk.Combobox(self.settings_frame, textvariable=self.batch_var, 
                                  values=["1", "4", "8", "16", "32"], width=8, state="readonly")
        batch_combo.grid(row=2, column=1, sticky=tk.W, pady=(5, 0))
        
        # Precision
        ttk.Label(self.settings_frame, text="Precision:").grid(row=3, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        self.precision_var = tk.StringVar(value="FP32")
        precision_combo = ttk.Combobox(self.settings_frame, textvariable=self.precision_var,
                                     values=["FP32", "FP16"], width=8, state="readonly")
        precision_combo.grid(row=3, column=1, sticky=tk.W, pady=(5, 0))
        
        # Offline mode
        self.offline_var = tk.BooleanVar(value=False)
        offline_check = ttk.Checkbutton(self.settings_frame, text="Offline mode (no pretrained weights)", 
                                       variable=self.offline_var)
        offline_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Simple mode description
        self.simple_desc_frame = ttk.LabelFrame(main_frame, text="Quick Benchmark", padding="15")
        self.simple_desc_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        desc_text = ("This tool will benchmark your hardware using three different AI models:\n"
                    "• ResNet50 (Traditional CNN)\n"
                    "• EfficientNet-B0 (Modern efficient architecture)\n"
                    "• Vision Transformer (Attention-based model)\n\n"
                    "Click 'Run Benchmark' to start with default settings, or enable 'Advanced Mode' for customization.")
        
        desc_label = ttk.Label(self.simple_desc_frame, text=desc_text, justify=tk.LEFT)
        desc_label.pack()
        # Device info
        device_frame = ttk.LabelFrame(main_frame, text="Device Information", padding="10")
        device_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        device_frame.columnconfigure(1, weight=1)
        
        self.device_info_var = tk.StringVar()
        self.update_device_info()
        device_info_label = ttk.Label(device_frame, textvariable=self.device_info_var)
        device_info_label.grid(row=0, column=0, sticky=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.run_button = ttk.Button(button_frame, text="Run Benchmark", 
                                    command=self.run_benchmark, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop", 
                                     command=self.stop_benchmark, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(button_frame, text="🗑 Clear", 
                                      command=self.clear_output)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Advanced mode buttons
        self.save_button = ttk.Button(button_frame, text="💾 Save Results", 
                                     command=self.save_results)
        self.save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.test_button = ttk.Button(button_frame, text="Test Connection", 
                                     command=self.test_connection)
        self.test_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_button = ttk.Button(button_frame, text="Export CSV", 
                                       command=self.export_csv)
        self.export_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=6, column=0, columnspan=3, pady=(0, 5))
        
        # Output text area
        output_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        output_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=80)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Test model loading on startup
        self.test_models_on_startup()
    
    def update_device_info(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        info = f"Device: {device}"
        
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            info += f" ({gpu_name}, {memory:.1f}GB)"
        
        self.device_info_var.set(info)
        
    def log_output(self, message):
        """Thread-safe logging to output text area"""
        self.root.after(0, lambda: self._log_output_main_thread(message))
        
    def _log_output_main_thread(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_progress(self, value):
        """Thread-safe progress bar update"""
        self.root.after(0, lambda: self.progress_var.set(value))
        
    def update_status(self, status):
        """Thread-safe status update"""
        self.root.after(0, lambda: self.status_var.set(status))
        
    def run_benchmark(self):
        if self.is_running:
            return
            
        # Validate settings
        try:
            num_runs = int(self.runs_var.get())
            warmup_runs = int(self.warmup_var.get())
            batch_size = int(self.batch_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
            return
            
        # Check if at least one model is selected
        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one model")
            return
            
        self.is_running = True
        self.run_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress_var.set(0)
        self.results_data = {}
        
        # Start benchmark in separate thread
        self.benchmark_thread = threading.Thread(
            target=self._run_benchmark_thread, 
            args=(selected_models, num_runs, warmup_runs, batch_size),
            daemon=True
        )
        self.benchmark_thread.start()
        
    def _benchmark_single_model(self, model_name, model_info, batch_size, num_runs, warmup_runs, device, use_fp16):
        """Benchmark a single model"""
        try:
            self.update_status(f"Loading {model_name}...")
            if self.advanced_mode.get():
                self.log_output(f"\n{'='*60}")
                self.log_output(f"BENCHMARKING {model_name}")
                self.log_output(f"{'='*60}")
            else:
                self.log_output(f"\n Testing {model_name}...")
            
            # Load model
            model = model_info["model_fn"]().to(device)
            model.eval()
            
            # Convert to half precision if requested
            if use_fp16 and device.type == 'cuda':
                model = model.half()
                if self.advanced_mode.get():
                    self.log_output(f"Using FP16 precision")
            
            # Calculate theoretical FLOPS
            self.update_status(f"Calculating FLOPS for {model_name}...")
            input_res = model_info["input_size"]
            
            with torch.no_grad():
                # Create a single sample for FLOPS calculation
                sample_input = (1,) + input_res
                macs, params = get_model_complexity_info(
                    model, input_res, as_strings=False, print_per_layer_stat=False
                )
                flops_per_sample = macs * 2
                total_flops = flops_per_sample * batch_size
                
            # Display basic or detailed results based on mode
            if self.advanced_mode.get():
                self.log_output(f"Model: {model_name}")
                self.log_output(f"Input shape: {(batch_size,) + input_res}")
                self.log_output(f"Parameters: {params / 1e6:.2f}M")
                self.log_output(f"FLOPs per sample: {flops_per_sample / 1e9:.2f} GFLOPs")
                self.log_output(f"Total FLOPs (batch {batch_size}): {total_flops / 1e12:.2f} TFLOPs")
            else:
                self.log_output(f" Parameters: {params / 1e6:.1f}M | 🧮 FLOPs: {flops_per_sample / 1e9:.1f} GFLOPs")
            
            # Prepare input tensor
            input_tensor = torch.randn(batch_size, *input_res, device=device)
            if use_fp16 and device.type == 'cuda':
                input_tensor = input_tensor.half()
            
            # Warmup
            if warmup_runs > 0:
                self.update_status(f"Warming up {model_name}...")
                if self.advanced_mode.get():
                    self.log_output(f"Warming up with {warmup_runs} runs...")
                with torch.no_grad():
                    for i in range(warmup_runs):
                        if not self.is_running:
                            return None
                        _ = model(input_tensor)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()
            
            if not self.is_running:
                return None
                
            # Benchmark
            self.update_status(f"Benchmarking {model_name}...")
            if self.advanced_mode.get():
                self.log_output(f"Running {num_runs} iterations...")
            else:
                self.log_output(f" ⏱️ Running {num_runs} iterations...")
            
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                start = time.time()
                for i in range(num_runs):
                    if not self.is_running:
                        return None
                    _ = model(input_tensor)
                        
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                end = time.time()
                
            # Calculate results
            duration = end - start
            avg_time_per_batch = duration / num_runs
            avg_time_per_sample = avg_time_per_batch / batch_size
            samples_per_second = batch_size * num_runs / duration
            tflops_runtime = total_flops * num_runs / duration / 1e12
            utilization = tflops_runtime / (total_flops / 1e12) if total_flops > 0 else 0
            
            # Store results
            results = {
                'model': model_name,
                'batch_size': batch_size,
                'precision': 'FP16' if use_fp16 else 'FP32',
                'parameters_m': params / 1e6,
                'flops_per_sample_g': flops_per_sample / 1e9,
                'total_flops_t': total_flops / 1e12,
                'avg_time_per_batch_ms': avg_time_per_batch * 1000,
                'avg_time_per_sample_ms': avg_time_per_sample * 1000,
                'samples_per_second': samples_per_second,
                'runtime_tflops': tflops_runtime,
                'utilization_percent': utilization * 100
            }
            
            # Display results
            self.log_output(f"\nRESULTS:")
            self.log_output(f"Total time for {num_runs} batches: {duration:.3f}s")
            self.log_output(f"Average time per batch: {avg_time_per_batch*1000:.2f}ms")
            self.log_output(f"Average time per sample: {avg_time_per_sample*1000:.2f}ms")
            self.log_output(f"Throughput: {samples_per_second:.1f} samples/second")
            self.log_output(f"Runtime throughput: {tflops_runtime:.2f} TFLOPS")
            self.log_output(f"Hardware utilization: {utilization*100:.1f}%")
            
            return results
            
        except Exception as e:
            self.log_output(f"Error benchmarking {model_name}: {str(e)}")
            return None
        
    def _run_benchmark_thread(self, selected_models, num_runs, warmup_runs, batch_size):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            use_fp16 = self.precision_var.get() == "FP16" and device.type == 'cuda'
            
            if self.advanced_mode.get():
                self.log_output(f"Starting benchmark on {device}")
                self.log_output(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
                self.log_output(f"Batch size: {batch_size}")
                self.log_output(f"Runs: {num_runs}, Warmup: {warmup_runs}")
            else:
                self.log_output(f"Starting quick benchmark on {device}...")
                self.log_output(f"Testing {len(selected_models)} models with default settings...")
            
            total_models = len(selected_models)
            
            for i, model_name in enumerate(selected_models):
                if not self.is_running:
                    break
                    
                model_info = self.models[model_name]
                progress_start = (i / total_models) * 100
                progress_end = ((i + 1) / total_models) * 100
                self.update_progress(progress_start)
                
                results = self._benchmark_single_model(
                    model_name, model_info, batch_size, num_runs, warmup_runs, device, use_fp16
                )
                
                if results:
                    self.results_data[model_name] = results
                
                self.update_progress(progress_end)
                
            if self.is_running:
                self._display_summary()
                self.update_progress(100)
                self.update_status("Benchmark completed")
            
        except Exception as e:
            self.log_output(f"Benchmark error: {str(e)}")
            messagebox.showerror("Benchmark Error", f"An error occurred: {str(e)}")
        finally:
            self.root.after(0, self._reset_ui_state)
            
    def _display_summary(self):
        """Display benchmark summary"""
        if not self.results_data:
            return
            
        self.log_output(f"\n{'='*80}")
        self.log_output("BENCHMARK SUMMARY")
        self.log_output(f"{'='*80}")
        
        # Table header
        header = f"{'Model':<15} {'Params(M)':<10} {'GFLOP/s':<10} {'ms/sample':<12} {'samples/s':<12} {'TFLOPS':<10} {'Util%':<8}"
        self.log_output(header)
        self.log_output("-" * len(header))
        
        # Sort by throughput
        sorted_results = sorted(self.results_data.values(), key=lambda x: x['runtime_tflops'], reverse=True)
        
        for result in sorted_results:
            row = f"{result['model']:<15} {result['parameters_m']:<10.1f} {result['flops_per_sample_g']:<10.1f} " \
                  f"{result['avg_time_per_sample_ms']:<12.2f} {result['samples_per_second']:<12.1f} " \
                  f"{result['runtime_tflops']:<10.2f} {result['utilization_percent']:<8.1f}"
            self.log_output(row)
            
        # Best performer
        best = max(sorted_results, key=lambda x: x['runtime_tflops'])
        self.log_output(f"\nBest performer: {best['model']} ({best['runtime_tflops']:.2f} TFLOPS)")
        
    def _reset_ui_state(self):
        self.is_running = False
        self.run_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.update_status("Ready")
        
    def test_connection(self):
        """Test internet connectivity for model downloads"""
        def test_thread():
            try:
                import urllib.request
                self.log_output("Testing internet connectivity...")
                
                # Test PyTorch model hub connectivity
                test_url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
                urllib.request.urlopen(test_url, timeout=10)
                self.log_output("✓ Internet connection OK - pretrained weights available")
                
            except Exception as e:
                self.log_output(f"✗ Internet connection failed: {str(e)}")
                self.log_output("Recommendation: Enable 'Offline mode' for benchmark without pretrained weights")
                
        threading.Thread(target=test_thread, daemon=True).start()
        
    def stop_benchmark(self):
        self.is_running = False
        self.log_output("Stopping benchmark...")
        self.update_status("Stopping...")
        
    def clear_output(self):
        self.output_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        self.results_data = {}
        self.update_status("Ready")
        
    def save_results(self):
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.output_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
            
    def export_csv(self):
        """Export results to CSV"""
        if not self.results_data:
            messagebox.showwarning("Warning", "No results to export")
            return
            
        try:
            from tkinter import filedialog
            import csv
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        'Model', 'Batch Size', 'Precision', 'Parameters (M)', 
                        'GFLOP/sample', 'Total TFLOP', 'ms/batch', 'ms/sample',
                        'samples/s', 'Runtime TFLOPS', 'Utilization %'
                    ])
                    
                    # Data
                    for result in self.results_data.values():
                        writer.writerow([
                            result['model'], result['batch_size'], result['precision'],
                            f"{result['parameters_m']:.2f}", f"{result['flops_per_sample_g']:.2f}",
                            f"{result['total_flops_t']:.2f}", f"{result['avg_time_per_batch_ms']:.2f}",
                            f"{result['avg_time_per_sample_ms']:.2f}", f"{result['samples_per_second']:.1f}",
                            f"{result['runtime_tflops']:.2f}", f"{result['utilization_percent']:.1f}"
                        ])
                        
                messagebox.showinfo("Success", f"Results exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {str(e)}")

def main():
    root = tk.Tk()
    
    # Set application style
    try:
        # Try to use modern theme if available
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
    except:
        pass
        
    app = ModelBenchmarkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()