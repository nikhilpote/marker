#!/usr/bin/env python3
"""
Marker Output Visualizer

Takes the JSON output from Marker and creates an HTML visualization
with bounding boxes drawn around each content block.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


class MarkerVisualizer:
    """Visualize Marker's JSON output with bounding boxes"""
    
    def __init__(self, json_path: str, output_path: str = None):
        """
        Initialize the visualizer with a JSON file path
        
        Args:
            json_path: Path to Marker's JSON output file
            output_path: Path to save the HTML visualization (auto-generated if None)
        """
        self.json_path = json_path
        if output_path is None:
            # Auto-generate output path by replacing .json with _visualization.html
            self.output_path = str(Path(json_path).with_suffix('')) + '_visualization.html'
        else:
            self.output_path = output_path
    
    def load_json(self) -> Dict[str, Any]:
        """Load the JSON data from the file"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def collect_blocks(self, node: Dict, page_id: int) -> List[Dict]:
        """
        Recursively collect all blocks with bbox and html from the JSON tree
        
        Args:
            node: Current node in the JSON tree
            page_id: Page identifier
            
        Returns:
            List of blocks with their properties
        """
        blocks = []
        
        # Check if this node has children
        has_children = node.get('children') and len(node['children']) > 0
        
        # Recursively collect from children
        if has_children:
            for child in node['children']:
                blocks.extend(self.collect_blocks(child, page_id))
        else:
            # Leaf node - include if it has bbox and html
            if node.get('bbox') and node.get('html'):
                blocks.append({
                    'bbox': node['bbox'],
                    'html': node['html'],
                    'block_type': node.get('block_type', 'Unknown'),
                    'id': node.get('id', ''),
                    'page_id': page_id
                })
        
        return blocks
    
    def generate_html(self) -> str:
        """Generate the HTML visualization"""
        json_data = self.load_json()
        
        # Extract all blocks
        all_blocks = []
        if isinstance(json_data, dict) and 'children' in json_data:
            for page_idx, page in enumerate(json_data['children']):
                if page.get('block_type') == 'Page':
                    blocks = self.collect_blocks(page, page_idx)
                    all_blocks.extend(blocks)
        
        # Group blocks by page
        pages = {}
        for block in all_blocks:
            page_id = block['page_id']
            if page_id not in pages:
                pages[page_id] = []
            pages[page_id].append(block)
        
        # Generate HTML template
        html_content = self._get_html_template()
        
        # Generate HTML for each page
        for page_id in sorted(pages.keys()):
            html_content += self._generate_page_html(page_id, pages[page_id])
        
        html_content += self._get_html_footer()
        
        return html_content
    
    def _get_html_template(self) -> str:
        """Get the HTML template header"""
        return """<!DOCTYPE html>
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
    
    def _generate_page_html(self, page_id: int, page_blocks: List[Dict]) -> str:
        """Generate HTML for a single page"""
        html = f'<div class="page-container"><div class="page-header">Page {page_id + 1}</div>'
        
        if page_blocks:
            # Get max coordinates to determine page size
            max_x = max(b['bbox'][2] for b in page_blocks)
            max_y = max(b['bbox'][3] for b in page_blocks)
            scale = 800 / max_x if max_x > 800 else 1  # Scale to fit 800px wide
            
            html += f'<div style="position: relative; width: {max_x * scale}px; margin: 0 auto;">'
            
            for block in page_blocks:
                bbox = block['bbox']
                x, y, x1, y1 = bbox
                width = x1 - x
                height = y1 - y
                
                # Escape HTML in content
                content = block['html'].replace('<', '&lt;').replace('>', '&gt;')[:100]
                if len(block['html']) > 100:
                    content += '...'
                
                html += f'''
                <div class="block" style="left: {x * scale}px; top: {y * scale}px; width: {width * scale}px; height: {height * scale}px;">
                    <div class="block-label">{block['block_type']}</div>
                    <div class="block-content">{content}</div>
                </div>
                '''
            
            html += '</div>'
        
        html += '</div>'
        return html
    
    def _get_html_footer(self) -> str:
        """Get the HTML template footer"""
        return """
</body>
</html>
"""
    
    def render(self):
        """Generate and save the HTML visualization"""
        html_content = self.generate_html()
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML visualization saved to: {self.output_path}")
        return self.output_path


def main():
    """Command-line interface"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python marker_visualizer.py <json_file> [output_file]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(json_path):
        print(f"Error: File '{json_path}' not found")
        sys.exit(1)
    
    visualizer = MarkerVisualizer(json_path, output_path)
    visualizer.render()


if __name__ == "__main__":
    main()

