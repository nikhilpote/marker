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
import json
from marker.config.parser import ConfigParser
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output
from marker.settings import settings

configure_logging()
logger = get_logger()

def generate_html_with_boxes(json_data, output_path):
    """Generate HTML with content rendered using coordinates and boxes"""
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Marker Output with Bounding Boxes</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background: #f5f5f5;
        }
        .page-container {
            background: white;
            margin: 20px auto;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .page-header {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
        }
        .block {
            position: absolute;
            border: 1px solid rgba(0,123,255,0.3);
            background: rgba(0,123,255,0.05);
            padding: 2px;
            margin: 1px;
        }
        .block-label {
            position: absolute;
            top: -12px;
            left: 0;
            background: rgba(0,123,255,0.9);
            color: white;
            font-size: 9px;
            padding: 1px 3px;
            border-radius: 2px;
            font-weight: bold;
        }
        .block-content {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 11px;
            padding: 1px;
        }
    </style>
</head>
<body>
"""
    
    # Extract blocks and pages from JSON
    def collect_blocks(node, page_id):
        blocks = []
        # Always recursively collect from children
        if node.get('children'):
            for child in node['children']:
                blocks.extend(collect_blocks(child, page_id))
        
        # Also include this node if it has bbox and html (leaf node with content)
        if node.get('bbox') and node.get('html'):
            blocks.append({
                'bbox': node['bbox'],
                'html': node['html'],
                'block_type': node.get('block_type', 'Unknown'),
                'id': node.get('id', ''),
                'page_id': page_id
            })
        return blocks
    
    all_blocks = []
    if isinstance(json_data, dict) and 'children' in json_data:
        for page_idx, page in enumerate(json_data['children']):
            if page.get('block_type') == 'Page':
                blocks = collect_blocks(page, page_idx)
                all_blocks.extend(blocks)
    
    # Group blocks by page
    pages = {}
    for block in all_blocks:
        page_id = block['page_id']
        if page_id not in pages:
            pages[page_id] = []
        pages[page_id].append(block)
    
    # Generate HTML for each page
    for page_id in sorted(pages.keys()):
        html_template += f'<div class="page-container"><div class="page-header">Page {page_id + 1}</div>'
        
        page_blocks = pages[page_id]
        # Find page dimensions to scale
        if page_blocks:
            # Get max coordinates to determine page size
            max_x = max(b['bbox'][2] for b in page_blocks)
            max_y = max(b['bbox'][3] for b in page_blocks)
            scale = 800 / max_x if max_x > 800 else 1  # Scale to fit 800px wide
            
            html_template += f'<div style="position: relative; width: {max_x * scale}px; margin: 0 auto;">'
            
            for block in page_blocks:
                bbox = block['bbox']
                x, y, x1, y1 = bbox
                width = x1 - x
                height = y1 - y
                
                # Escape HTML in content
                content = block['html'].replace('<', '&lt;').replace('>', '&gt;')[:100]
                if len(block['html']) > 100:
                    content += '...'
                
                html_template += f'''
                <div class="block" style="left: {x * scale}px; top: {y * scale}px; width: {width * scale}px; height: {height * scale}px;">
                    <div class="block-label">{block['block_type']}</div>
                    <div class="block-content">{content}</div>
                </div>
                '''
            
            html_template += '</div>'
        html_template += '</div>'
    
    html_template += """
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f"✓ HTML visualization saved to: {output_path}")

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
            "output_format": "json",  # Use JSON to get coordinates
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
        
        # Generate HTML visualization with boxes
        json_data = rendered.model_dump(exclude=["metadata"])
        html_path = os.path.join(out_folder, f"{config_parser.get_base_filename(pdf_file)}_visualization.html")
        generate_html_with_boxes(json_data, html_path)
        
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

