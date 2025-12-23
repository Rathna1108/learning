"""
Custom Document Layout Analysis System
Pure Computer Vision approach - No AI models
Uses OpenCV morphological operations and contour analysis
Highly customizable for specific document types
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict

class CustomLayoutAnalyzer:
    def __init__(self, config=None):
        """
        Initialize with customizable parameters
        
        Args:
            config: Dictionary of configuration parameters
        """
        # Default configuration - easily customizable
        self.config = {
            # Preprocessing
            'denoise_h': 10,
            'denoise_template_window': 7,
            'denoise_search_window': 21,
            
            # Morphological operations
            'dilate_kernel_width': 25,   # Horizontal dilation to connect words
            'dilate_kernel_height': 3,    # Vertical dilation to connect lines
            'dilate_iterations': 2,
            
            # Contour filtering
            'min_width': 60,
            'min_height': 20,
            'min_area': 1500,
            'max_aspect_ratio': 20,
            'min_aspect_ratio': 0.1,
            
            # Column detection
            'column_gap_threshold': 60,   # Minimum gap between columns (pixels)
            
            # Text region classification
            'header_zone': 0.08,          # Top 8% is header zone
            'footer_zone': 0.92,          # Bottom 8% is footer zone
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        print("✅ Custom Layout Analyzer initialized")
        print(f"   Config: {self.config}\n")
    
    def analyze_document(self, image_path, visualize=True):
        """
        Main analysis pipeline
        
        Steps:
        1. Load and preprocess image
        2. Detect text regions using morphology
        3. Filter and clean regions
        4. Detect columns
        5. Sort reading order
        6. Classify regions
        7. Visualize results
        """
        print(f"📄 Analyzing: {image_path}")
        
        # Step 1: Load and preprocess
        image, binary, processed = self._preprocess_image(image_path)
        h, w = image.shape[:2]
        
        # Step 2: Detect text regions
        contours = self._detect_text_regions(processed)
        print(f"   Found {len(contours)} initial contours")
        
        # Step 3: Convert to bounding boxes and filter
        boxes = self._contours_to_boxes(contours, w, h)
        print(f"   Converted to {len(boxes)} bounding boxes")
        
        # Step 4: Merge overlapping boxes
        merged_boxes = self._merge_overlapping_boxes(boxes)
        print(f"   Merged to {len(merged_boxes)} regions")
        
        # Step 5: Filter by size and shape
        filtered_boxes = self._filter_boxes(merged_boxes, w, h)
        print(f"   Filtered to {len(filtered_boxes)} valid regions")
        
        # Step 6: Detect columns and sort
        sorted_boxes = self._sort_reading_order(filtered_boxes, w, h)
        
        # Step 7: Classify regions
        classified_boxes = self._classify_regions(sorted_boxes, w, h)
        
        # Step 8: Visualize
        if visualize:
            self._visualize_results(image, classified_boxes)
        
        return {
            'boxes': classified_boxes,
            'image_size': (w, h),
            'num_regions': len(classified_boxes)
        }
    
    def _preprocess_image(self, image_path):
        """Step 1: Load and preprocess image"""
        print("🔄 Preprocessing image...")
        
        # Load image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=self.config['denoise_h'],
            templateWindowSize=self.config['denoise_template_window'],
            searchWindowSize=self.config['denoise_search_window']
        )
        
        # Adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Inverted: text is white, background is black
            11,
            2
        )
        
        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.config['dilate_kernel_width'], 
             self.config['dilate_kernel_height'])
        )
        
        dilated = cv2.dilate(
            binary,
            kernel,
            iterations=self.config['dilate_iterations']
        )
        
        print("   ✓ Denoised, binarized, and dilated")
        
        return image, binary, dilated
    
    def _detect_text_regions(self, processed_image):
        """Step 2: Detect text regions using contours"""
        print("🔍 Detecting text regions...")
        
        # Find contours
        contours, _ = cv2.findContours(
            processed_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        return contours
    
    def _contours_to_boxes(self, contours, img_width, img_height):
        """Step 3: Convert contours to bounding boxes"""
        boxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Basic size filter (remove very tiny noise)
            if w > 10 and h > 5:
                boxes.append({
                    'bbox': (x, y, x + w, y + h),
                    'width': w,
                    'height': h,
                    'area': w * h,
                    'x_center': x + w / 2,
                    'y_center': y + h / 2
                })
        
        return boxes
    
    def _merge_overlapping_boxes(self, boxes):
        """Step 4: Merge boxes that overlap significantly"""
        if not boxes:
            return []
        
        # Sort by Y coordinate
        boxes = sorted(boxes, key=lambda b: b['bbox'][1])
        
        merged = []
        current = boxes[0].copy()
        
        for i in range(1, len(boxes)):
            box = boxes[i]
            
            # Check if boxes overlap
            x1_curr, y1_curr, x2_curr, y2_curr = current['bbox']
            x1_box, y1_box, x2_box, y2_box = box['bbox']
            
            # Calculate overlap
            x_overlap = max(0, min(x2_curr, x2_box) - max(x1_curr, x1_box))
            y_overlap = max(0, min(y2_curr, y2_box) - max(y1_curr, y1_box))
            
            overlap_area = x_overlap * y_overlap
            min_area = min(current['area'], box['area'])
            
            # If significant overlap (>30%), merge
            if overlap_area > 0.3 * min_area:
                # Merge boxes
                new_x1 = min(x1_curr, x1_box)
                new_y1 = min(y1_curr, y1_box)
                new_x2 = max(x2_curr, x2_box)
                new_y2 = max(y2_curr, y2_box)
                
                current = {
                    'bbox': (new_x1, new_y1, new_x2, new_y2),
                    'width': new_x2 - new_x1,
                    'height': new_y2 - new_y1,
                    'area': (new_x2 - new_x1) * (new_y2 - new_y1),
                    'x_center': (new_x1 + new_x2) / 2,
                    'y_center': (new_y1 + new_y2) / 2
                }
            else:
                # No overlap, save current and start new
                merged.append(current)
                current = box.copy()
        
        # Don't forget last box
        merged.append(current)
        
        return merged
    
    def _filter_boxes(self, boxes, img_width, img_height):
        """Step 5: Filter boxes by size and shape criteria"""
        filtered = []
        
        for box in boxes:
            w = box['width']
            h = box['height']
            area = box['area']
            
            # Size filters
            if w < self.config['min_width']:
                continue
            if h < self.config['min_height']:
                continue
            if area < self.config['min_area']:
                continue
            
            # Aspect ratio filter (remove very wide/tall artifacts)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > self.config['max_aspect_ratio']:
                continue
            if aspect_ratio < self.config['min_aspect_ratio']:
                continue
            
            filtered.append(box)
        
        return filtered
    
    def _sort_reading_order(self, boxes, img_width, img_height):
        """Step 6: Detect columns and sort in reading order"""
        print("📊 Detecting columns and sorting...")
        
        if not boxes:
            return []
        
        # Extract X centers
        x_centers = [box['x_center'] for box in boxes]
        
        if len(x_centers) < 2:
            # Single column - sort by Y
            boxes.sort(key=lambda b: b['bbox'][1])
            print(f"   Single column layout")
            return boxes
        
        # Find column boundary using gap detection
        sorted_x = sorted(x_centers)
        max_gap = 0
        column_boundary = img_width / 2
        
        for i in range(len(sorted_x) - 1):
            gap = sorted_x[i + 1] - sorted_x[i]
            if gap > max_gap:
                max_gap = gap
                column_boundary = (sorted_x[i] + sorted_x[i + 1]) / 2
        
        # Check if gap is significant
        if max_gap < self.config['column_gap_threshold']:
            # Single column
            boxes.sort(key=lambda b: b['bbox'][1])
            print(f"   Single column layout")
            return boxes
        
        # Two-column layout
        left_col = [b for b in boxes if b['x_center'] < column_boundary]
        right_col = [b for b in boxes if b['x_center'] >= column_boundary]
        
        # Sort each column by Y
        left_col.sort(key=lambda b: b['bbox'][1])
        right_col.sort(key=lambda b: b['bbox'][1])
        
        print(f"   2-column layout: Left={len(left_col)}, Right={len(right_col)}")
        print(f"   Column boundary at X={int(column_boundary)}")
        
        # Combine: left first, then right
        return left_col + right_col
    
    def _classify_regions(self, boxes, img_width, img_height):
        """Step 7: Classify regions based on position and size"""
        print("🏷️  Classifying regions...")
        
        classified = []
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box['bbox']
            
            # Default classification
            region_type = 'content'
            
            # Header detection (top zone)
            if y1 < img_height * self.config['header_zone']:
                region_type = 'header'
            
            # Footer detection (bottom zone)
            elif y1 > img_height * self.config['footer_zone']:
                region_type = 'footer'
            
            # Page number (small box in corners)
            elif (box['area'] < 5000 and 
                  (x1 < img_width * 0.15 or x2 > img_width * 0.85)):
                region_type = 'page_number'
            
            # Title (centered, upper portion, not too large)
            elif (abs(box['x_center'] - img_width / 2) < img_width * 0.2 and
                  y1 < img_height * 0.3 and
                  box['area'] < 50000):
                region_type = 'title'
            
            classified.append({
                'id': idx,
                'type': region_type,
                'bbox': box['bbox'],
                'area': box['area'],
                'reading_order': idx + 1
            })
        
        # Count types
        type_counts = defaultdict(int)
        for box in classified:
            type_counts[box['type']] += 1
        
        print(f"   Classification: {dict(type_counts)}")
        
        return classified
    
    def _visualize_results(self, image, boxes):
        """Step 8: Visualize detected regions"""
        print("🎨 Creating visualization...")
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 18))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Color map for region types
        color_map = {
            'header': 'red',
            'title': 'purple',
            'content': 'green',
            'footer': 'orange',
            'page_number': 'yellow'
        }
        
        # Rainbow colors for reading order
        colors = plt.cm.rainbow(np.linspace(0, 1, len(boxes)))
        
        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            region_type = box['type']
            reading_order = box['reading_order']
            
            # Draw rectangle
            color = colors[box['id']]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5,
                edgecolor=color,
                facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)
            
            # Add reading order number
            ax.text(
                x1 + 5, y1 + 25, str(reading_order),
                color='white',
                fontsize=14,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=color,
                    edgecolor='white',
                    linewidth=2,
                    alpha=0.9
                )
            )
            
            # Add region type label
            ax.text(
                x1 + 5, y2 - 10, region_type.upper(),
                color='white',
                fontsize=9,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color_map.get(region_type, 'blue'),
                    alpha=0.7
                )
            )
        
        ax.axis('off')
        plt.title(
            f'Custom Layout Analysis - {len(boxes)} Regions\n'
            f'Numbers show reading order | Colors indicate type',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        plt.tight_layout()
        
        # Save
        output_file = 'custom_layout_visualization.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        
        plt.show()
    
    def print_results(self, result):
        """Pretty print analysis results"""
        print("\n" + "="*70)
        print("📊 LAYOUT ANALYSIS RESULTS")
        print("="*70)
        
        boxes = result['boxes']
        img_w, img_h = result['image_size']
        
        print(f"\n📐 Image size: {img_w} × {img_h}")
        print(f"📦 Total regions: {result['num_regions']}\n")
        
        # Group by type
        by_type = defaultdict(list)
        for box in boxes:
            by_type[box['type']].append(box)
        
        for region_type, regions in sorted(by_type.items()):
            print(f"\n{'='*70}")
            print(f"🏷️  {region_type.upper()} ({len(regions)} regions)")
            print(f"{'='*70}")
            
            for box in regions:
                x1, y1, x2, y2 = box['bbox']
                print(f"  [{box['reading_order']}] Position: ({x1}, {y1}) → ({x2}, {y2})")
                print(f"      Size: {x2-x1} × {y2-y1} | Area: {box['area']}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create analyzer with default config
    analyzer = CustomLayoutAnalyzer()
    
    # Or customize the configuration
    # custom_config = {
    #     'dilate_kernel_width': 30,  # More aggressive horizontal connection
    #     'min_width': 80,            # Stricter size filter
    #     'column_gap_threshold': 80  # Wider column gap needed
    # }
    # analyzer = CustomLayoutAnalyzer(config=custom_config)
    
    # Analyze document
    image_path = "images/Dhingra ENT 8th Edition_Split_page-0007.jpg"  # ⚠️ CHANGE THIS
    
    try:
        result = analyzer.analyze_document(
            image_path=image_path,
            visualize=True
        )
        
        # Print detailed results
        analyzer.print_results(result)
        
        # Access boxes programmatically
        print("\n🎯 Programmatic Access Example:")
        for box in result['boxes'][:3]:  # First 3 boxes
            print(f"  Box {box['reading_order']}: {box['type']} at {box['bbox']}")
        
    except FileNotFoundError:
        print(f"\n❌ Image not found: '{image_path}'")
        print("   Update image_path with your file")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()