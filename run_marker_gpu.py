#!/usr/bin/env python3
"""
Run Marker with GPU support
"""
import os
import sys
import time
from pathlib import Path

# Set GPU environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_DEVICE"] = "cuda"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from marker.config.parser import ConfigParser
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output
from marker.settings import settings

configure_logging()
logger = get_logger()

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_marker_gpu.py <pdf_file>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    if not Path(pdf_file).exists():
        print(f"Error: File '{pdf_file}' not found")
        sys.exit(1)
    
    print(f"Converting {pdf_file} using GPU...")
    print(f"Torch device: {settings.TORCH_DEVICE_MODEL}")
    
    try:
        start = time.time()
        
        # Create models
        models = create_model_dict()
        
        # Parse config with default options and reduced batch sizes for limited GPU memory
        config_parser = ConfigParser({
            "output_format": "json",  # JSON needed for coordinates and full reconstruction
            "output_dir": "./output",
            "layout_batch_size": 1,
            "detection_batch_size": 1,
            "table_rec_batch_size": 1,
            "ocr_error_batch_size": 1,
            "recognition_batch_size": 4,
            "equation_batch_size": 1,
        })
        
        # Get converter
        converter_cls = config_parser.get_converter_cls()
        converter = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        
        # Convert
        rendered = converter(pdf_file)
        
        # Save output
        out_folder = config_parser.get_output_folder(pdf_file)
        save_output(rendered, out_folder, config_parser.get_base_filename(pdf_file))
        
        total_time = time.time() - start
        print(f"✓ Conversion complete!")
        print(f"✓ Output saved to: {out_folder}")
        print(f"✓ Total time: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"✓ GPU memory cleared")

if __name__ == "__main__":
    main()

