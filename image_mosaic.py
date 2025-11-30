#!/usr/bin/env python3
"""
Advanced Image Mosaic Generator

A powerful command-line tool that creates stunning image mosaics from a collection of tile images.
Features multiple color matching algorithms, tile rotation, blending modes, and more.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import math
from collections import Counter, defaultdict
from enum import Enum

try:
    from PIL import Image, ImageEnhance, ImageDraw, ImageStat
except ImportError:
    print("Error: Pillow is not installed. Please run: pip install Pillow")
    sys.exit(1)


class ColorMatchMethod(Enum):
    """Color matching algorithm options."""
    RGB_EUCLIDEAN = "rgb"
    LAB_PERCEPTUAL = "lab"
    WEIGHTED_RGB = "weighted"


class BlendMode(Enum):
    """Blending mode options."""
    NONE = "none"
    OVERLAY = "overlay"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    SOFT_LIGHT = "soft_light"


def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert RGB to LAB color space for perceptual color matching.
    
    Args:
        r, g, b: RGB values (0-255)
        
    Returns:
        LAB color tuple (L, A, B)
    """
    # Normalize RGB to 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    # Convert to linear RGB
    def f(t):
        return t / 12.92 if t <= 0.04045 else ((t + 0.055) / 1.055) ** 2.4
    
    r, g, b = f(r), f(g), f(b)
    
    # Convert to XYZ
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    
    # Convert to LAB
    def f_inv(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)
    
    fx, fy, fz = f_inv(x), f_inv(y), f_inv(z)
    L = 116 * fy - 16
    A = 500 * (fx - fy)
    B = 200 * (fy - fz)
    
    return (L, A, B)


def get_average_color(image: Image.Image, method: str = "center") -> Tuple[int, int, int]:
    """
    Calculate the average RGB color of an image using different sampling methods.
    
    Args:
        image: PIL Image object
        method: Sampling method - "average" (all pixels), "center" (center region), "dominant" (most common color)
        
    Returns:
        Tuple of (R, G, B) average color values
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if method == "center":
        # Sample center 50% of the image
        w, h = image.size
        x1, y1 = int(w * 0.25), int(h * 0.25)
        x2, y2 = int(w * 0.75), int(h * 0.75)
        region = image.crop((x1, y1, x2, y2))
        pixels = list(region.getdata())
    elif method == "dominant":
        # Get most common color (simplified - uses quantized colors)
        pixels = list(image.getdata())
        # Quantize to reduce color space
        quantized = [(r//32*32, g//32*32, b//32*32) for r, g, b in pixels]
        color_counts = Counter(quantized)
        return color_counts.most_common(1)[0][0]
    else:  # average
        pixels = list(image.getdata())
    
    if not pixels:
        return (0, 0, 0)
    
    r_sum = g_sum = b_sum = 0
    for r, g, b in pixels:
        r_sum += r
        g_sum += g
        b_sum += b
    
    total = len(pixels)
    return (r_sum // total, g_sum // total, b_sum // total)


def color_distance(color1: Tuple[int, int, int], 
                   color2: Tuple[int, int, int], 
                   method: ColorMatchMethod = ColorMatchMethod.RGB_EUCLIDEAN) -> float:
    """
    Calculate distance between two RGB colors using different algorithms.
    
    Args:
        color1: First RGB color tuple
        color2: Second RGB color tuple
        method: Color matching method to use
        
    Returns:
        Color distance value (lower = more similar)
    """
    if method == ColorMatchMethod.RGB_EUCLIDEAN:
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)
    
    elif method == ColorMatchMethod.WEIGHTED_RGB:
        # Weighted RGB (human eye is more sensitive to green)
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return math.sqrt(2 * (r1 - r2) ** 2 + 4 * (g1 - g2) ** 2 + 3 * (b1 - b2) ** 2)
    
    elif method == ColorMatchMethod.LAB_PERCEPTUAL:
        # LAB color space for perceptual matching
        lab1 = rgb_to_lab(*color1)
        lab2 = rgb_to_lab(*color2)
        return math.sqrt((lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2)
    
    return float('inf')


def enhance_image(image: Image.Image, 
                  brightness: float = 1.0,
                  contrast: float = 1.0,
                  saturation: float = 1.0) -> Image.Image:
    """
    Apply color enhancements to an image.
    
    Args:
        image: PIL Image object
        brightness: Brightness multiplier (1.0 = no change)
        contrast: Contrast multiplier (1.0 = no change)
        saturation: Saturation multiplier (1.0 = no change)
        
    Returns:
        Enhanced image
    """
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
    
    return image


def blend_images(base: Image.Image, overlay: Image.Image, mode: BlendMode, opacity: float = 0.5) -> Image.Image:
    """
    Blend two images together using various blending modes.
    
    Args:
        base: Base image
        overlay: Overlay image
        mode: Blending mode
        opacity: Opacity of overlay (0.0-1.0)
        
    Returns:
        Blended image
    """
    if mode == BlendMode.NONE or opacity == 0.0:
        return base
    
    if base.size != overlay.size:
        overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
    
    if mode == BlendMode.OVERLAY:
        # Simple overlay blend
        result = Image.blend(base, overlay, opacity)
    elif mode == BlendMode.MULTIPLY:
        # Multiply blend
        base_arr = base.load()
        overlay_arr = overlay.load()
        result = base.copy()
        result_arr = result.load()
        for y in range(base.height):
            for x in range(base.width):
                r1, g1, b1 = base_arr[x, y]
                r2, g2, b2 = overlay_arr[x, y]
                r = int(r1 * r2 / 255 * (1 - opacity) + r1 * opacity)
                g = int(g1 * g2 / 255 * (1 - opacity) + g1 * opacity)
                b = int(b1 * b2 / 255 * (1 - opacity) + b1 * opacity)
                result_arr[x, y] = (r, g, b)
    elif mode == BlendMode.SCREEN:
        # Screen blend
        base_arr = base.load()
        overlay_arr = overlay.load()
        result = base.copy()
        result_arr = result.load()
        for y in range(base.height):
            for x in range(base.width):
                r1, g1, b1 = base_arr[x, y]
                r2, g2, b2 = overlay_arr[x, y]
                r = int((255 - (255 - r1) * (255 - r2) / 255) * opacity + r1 * (1 - opacity))
                g = int((255 - (255 - g1) * (255 - g2) / 255) * opacity + g1 * (1 - opacity))
                b = int((255 - (255 - b1) * (255 - b2) / 255) * opacity + b1 * (1 - opacity))
                result_arr[x, y] = (r, g, b)
    else:  # SOFT_LIGHT
        result = Image.blend(base, overlay, opacity * 0.5)
    
    return result


def load_tile_images(tile_folder: str, 
                     enable_rotation: bool = False,
                     color_method: str = "average") -> List[Tuple[Image.Image, Tuple[int, int, int], int]]:
    """
    Load all images from a folder and calculate their average colors.
    Optionally creates rotated versions of tiles.
    
    Args:
        tile_folder: Path to folder containing tile images
        enable_rotation: If True, creates 90, 180, 270 degree rotated versions
        color_method: Method for color calculation ("average", "center", "dominant")
        
    Returns:
        List of tuples: (image, average_color, rotation_angle)
    """
    tile_folder_path = Path(tile_folder)
    if not tile_folder_path.exists():
        raise FileNotFoundError(f"Tile folder not found: {tile_folder}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    tiles = []
    image_files = [f for f in tile_folder_path.rglob('*') 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No image files found in {tile_folder} or its subfolders")
    
    print(f"Found {len(image_files)} tile images (scanning recursively)...")
    print(f"Loading and processing tiles...")
    
    rotations = [0] if not enable_rotation else [0, 90, 180, 270]
    total_to_process = len(image_files) * len(rotations)
    processed = 0
    
    for idx, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            for angle in rotations:
                if angle == 0:
                    rotated_img = img
                else:
                    rotated_img = img.rotate(angle, expand=True)
                
                avg_color = get_average_color(rotated_img, method=color_method)
                tiles.append((rotated_img, avg_color, angle))
                processed += 1
                
                # Show progress every 10% or every 50 tiles, whichever is more frequent
                if processed % max(1, total_to_process // 20) == 0 or processed == total_to_process:
                    progress = (processed / total_to_process) * 100
                    print(f"Loading tiles: {processed}/{total_to_process} ({progress:.1f}%)", end='\r')
        except Exception as e:
            print(f"\nWarning: Could not load {img_path.name}: {e}")
            continue
    
    print()  # New line after progress
    
    if not tiles:
        raise ValueError("No valid tile images could be loaded")
    
    print(f"Successfully loaded {len(tiles)} tile images (with rotations: {len(rotations)})")
    return tiles


def find_best_tile(target_color: Tuple[int, int, int], 
                   tiles: List[Tuple[Image.Image, Tuple[int, int, int], int]],
                   method: ColorMatchMethod = ColorMatchMethod.RGB_EUCLIDEAN,
                   used_tiles: Optional[Dict[int, int]] = None,
                   max_reuse: Optional[int] = None) -> Tuple[Image.Image, int]:
    """
    Find the best matching tile image, optionally limiting reuse.
    
    Args:
        target_color: Target RGB color
        tiles: List of (image, average_color, rotation) tuples
        method: Color matching method
        used_tiles: Dictionary tracking tile usage counts
        max_reuse: Maximum times a tile can be reused (None = unlimited)
        
    Returns:
        Tuple of (best_tile_image, tile_index)
    """
    best_tile = None
    best_index = -1
    best_distance = float('inf')
    
    for idx, (tile_img, tile_color, rotation) in enumerate(tiles):
        # Skip if tile has reached max reuse limit
        if max_reuse is not None and used_tiles is not None:
            if used_tiles.get(idx, 0) >= max_reuse:
                continue
        
        distance = color_distance(target_color, tile_color, method)
        if distance < best_distance:
            best_distance = distance
            best_tile = tile_img
            best_index = idx
    
    # Update usage count
    if used_tiles is not None and best_index >= 0:
        used_tiles[best_index] = used_tiles.get(best_index, 0) + 1
    
    return (best_tile, best_index)


def create_mosaic(target_image_path: str, 
                  tile_folder: str, 
                  output_path: str,
                  grid_size: Tuple[int, int] = None,
                  tile_size: Tuple[int, int] = None,
                  color_method: ColorMatchMethod = ColorMatchMethod.RGB_EUCLIDEAN,
                  enable_rotation: bool = False,
                  max_tile_reuse: Optional[int] = None,
                  blend_mode: BlendMode = BlendMode.NONE,
                  blend_opacity: float = 0.5,
                  brightness: float = 1.0,
                  contrast: float = 1.0,
                  saturation: float = 1.0,
                  border_width: int = 0,
                  border_color: Tuple[int, int, int] = (0, 0, 0),
                  sampling_method: str = "average",
                  show_stats: bool = False) -> Dict:
    """
    Create an advanced mosaic image from tile images with many customization options.
    
    Args:
        target_image_path: Path to the target image
        tile_folder: Path to folder containing tile images
        output_path: Path to save the output mosaic
        grid_size: Optional (rows, cols) tuple
        tile_size: Optional (width, height) tuple for each tile
        color_method: Color matching algorithm
        enable_rotation: Enable tile rotation for better matching
        max_tile_reuse: Maximum times a tile can be reused (None = unlimited)
        blend_mode: Blending mode with original image
        blend_opacity: Opacity for blending (0.0-1.0)
        brightness: Brightness adjustment for tiles (1.0 = no change)
        contrast: Contrast adjustment for tiles (1.0 = no change)
        saturation: Saturation adjustment for tiles (1.0 = no change)
        border_width: Width of border between tiles (0 = no border)
        border_color: RGB color of borders
        sampling_method: Method for sampling region colors ("average", "center", "dominant")
        show_stats: Whether to print statistics after completion
        
    Returns:
        Dictionary with statistics about the mosaic
    """
    # Load target image
    print(f"Loading target image: {target_image_path}")
    target_image = Image.open(target_image_path)
    if target_image.mode != 'RGB':
        target_image = target_image.convert('RGB')
    
    target_width, target_height = target_image.size
    print(f"Target image size: {target_width}x{target_height}")
    
    # Load tile images
    tiles = load_tile_images(tile_folder, enable_rotation=enable_rotation, color_method=sampling_method)
    
    # Determine grid size
    if grid_size is None:
        if tile_size:
            tile_w, tile_h = tile_size
            cols = target_width // tile_w
            rows = target_height // tile_h
        else:
            cols = max(20, target_width // 30)
            rows = max(20, target_height // 30)
    else:
        rows, cols = grid_size
    
    # Calculate actual tile size
    if tile_size:
        tile_width, tile_height = tile_size
    else:
        tile_width = target_width // cols
        tile_height = target_height // rows
    
    print(f"Creating mosaic with {rows}x{cols} grid ({rows * cols} tiles)")
    print(f"Each tile will be {tile_width}x{tile_height} pixels")
    print(f"Color matching: {color_method.value}, Rotation: {enable_rotation}, Max reuse: {max_tile_reuse}")
    
    # Create output image
    output_width = cols * tile_width + (cols - 1) * border_width if border_width > 0 else cols * tile_width
    output_height = rows * tile_height + (rows - 1) * border_width if border_width > 0 else rows * tile_height
    output_image = Image.new('RGB', (output_width, output_height), color=border_color)
    
    # Track statistics
    used_tiles = defaultdict(int)
    tile_usage = []
    
    # Process each grid cell
    total_cells = rows * cols
    processed = 0
    
    for row in range(rows):
        for col in range(cols):
            # Calculate region in target image
            x1 = int(col * target_width / cols)
            y1 = int(row * target_height / rows)
            x2 = int((col + 1) * target_width / cols)
            y2 = int((row + 1) * target_height / rows)
            
            # Extract region and get average color
            region = target_image.crop((x1, y1, x2, y2))
            region_color = get_average_color(region, method=sampling_method)
            
            # Find best matching tile
            best_tile, tile_idx = find_best_tile(
                region_color, tiles, method=color_method,
                used_tiles=used_tiles, max_reuse=max_tile_reuse
            )
            tile_usage.append(tile_idx)
            
            # Apply enhancements
            if brightness != 1.0 or contrast != 1.0 or saturation != 1.0:
                best_tile = enhance_image(best_tile, brightness, contrast, saturation)
            
            # Resize tile to fit
            resized_tile = best_tile.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            
            # Apply blending if enabled
            if blend_mode != BlendMode.NONE:
                region_resized = region.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                resized_tile = blend_images(region_resized, resized_tile, blend_mode, blend_opacity)
            
            # Calculate position (accounting for borders)
            output_x = col * (tile_width + border_width)
            output_y = row * (tile_height + border_width)
            
            # Paste tile into output image
            output_image.paste(resized_tile, (output_x, output_y))
            
            processed += 1
            if processed % max(1, total_cells // 20) == 0 or processed == total_cells:
                progress = (processed / total_cells) * 100
                bar_length = 40
                filled = int(bar_length * processed / total_cells)
                bar = '=' * filled + '-' * (bar_length - filled)
                print(f"Progress: [{bar}] {processed}/{total_cells} ({progress:.1f}%)", end='\r')
    
    print()  # New line after progress
    
    # Save output image
    print(f"Saving mosaic to: {output_path}")
    output_image.save(output_path, quality=95)
    print("Mosaic created successfully!")
    
    # Calculate and display statistics
    stats = {
        'total_tiles_used': len(set(tile_usage)),
        'unique_tiles_available': len(tiles),
        'total_cells': total_cells,
        'tile_reuse_counts': dict(used_tiles),
        'most_used_tile': max(used_tiles.items(), key=lambda x: x[1]) if used_tiles else None,
        'least_used_tile': min(used_tiles.items(), key=lambda x: x[1]) if used_tiles else None,
    }
    
    if show_stats:
        print("\n" + "="*50)
        print("MOSAIC STATISTICS")
        print("="*50)
        print(f"Total cells in mosaic: {stats['total_cells']}")
        print(f"Unique tiles available: {stats['unique_tiles_available']}")
        print(f"Unique tiles used: {stats['total_tiles_used']}")
        print(f"Tile usage rate: {stats['total_tiles_used']/stats['unique_tiles_available']*100:.1f}%")
        if stats['most_used_tile']:
            print(f"Most used tile (index {stats['most_used_tile'][0]}): {stats['most_used_tile'][1]} times")
        if stats['least_used_tile']:
            print(f"Least used tile (index {stats['least_used_tile'][0]}): {stats['least_used_tile'][1]} times")
        print("="*50)
    
    return stats


def batch_process(input_folder: str,
                  tile_folder: str,
                  output_folder: str,
                  **kwargs) -> None:
    """
    Process multiple target images in batch.
    
    Args:
        input_folder: Folder containing target images
        tile_folder: Folder containing tile images
        output_folder: Folder to save output mosaics
        **kwargs: Additional arguments passed to create_mosaic
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Processing {len(image_files)} images in batch mode...")
    
    for img_file in image_files:
        output_file = output_path / f"mosaic_{img_file.stem}{img_file.suffix}"
        print(f"\nProcessing: {img_file.name}")
        try:
            create_mosaic(
                target_image_path=str(img_file),
                tile_folder=tile_folder,
                output_path=str(output_file),
                **kwargs
            )
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            continue
    
    print(f"\nBatch processing complete! Output saved to: {output_folder}")


def main():
    """Main entry point for the command-line tool."""
    parser = argparse.ArgumentParser(
        description='Create advanced image mosaics from a collection of tile images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python image_mosaic.py target.jpg tiles/ output.jpg
  
  # High-quality mosaic with LAB color matching and rotation
  python image_mosaic.py target.jpg tiles/ output.jpg --color-method lab --rotation --grid 100 100
  
  # Mosaic with blending and borders
  python image_mosaic.py target.jpg tiles/ output.jpg --blend overlay --blend-opacity 0.3 --border 2
  
  # Limit tile reuse for more diversity
  python image_mosaic.py target.jpg tiles/ output.jpg --max-reuse 5
  
  # Enhanced colors with borders
  python image_mosaic.py target.jpg tiles/ output.jpg --brightness 1.2 --saturation 1.3 --border 1 --border-color 255 255 255
  
  # Batch process multiple images
  python image_mosaic.py --batch input_folder/ tiles/ output_folder/ --grid 50 50
        """
    )
    
    parser.add_argument('target_image', nargs='?',
                       help='Path to the target image (not needed in batch mode)')
    parser.add_argument('tile_folder', nargs='?',
                       help='Path to folder containing tile images')
    parser.add_argument('output', nargs='?',
                       help='Path to save the output mosaic image (or output folder in batch mode)')
    
    # Grid and size options
    parser.add_argument('--grid', 
                       type=int, 
                       nargs=2, 
                       metavar=('ROWS', 'COLS'),
                       help='Grid size (rows cols). If not specified, calculated automatically')
    parser.add_argument('--tile-size', 
                       type=int, 
                       nargs=2, 
                       metavar=('WIDTH', 'HEIGHT'),
                       dest='tile_size',
                       help='Size of each tile in pixels (width height)')
    
    # Color matching options
    parser.add_argument('--color-method',
                       choices=['rgb', 'lab', 'weighted'],
                       default='rgb',
                       help='Color matching algorithm: rgb (Euclidean), lab (perceptual), weighted (RGB with weights)')
    parser.add_argument('--sampling',
                       choices=['average', 'center', 'dominant'],
                       default='average',
                       help='Method for sampling region colors: average (all pixels), center (center region), dominant (most common)')
    
    # Tile options
    parser.add_argument('--rotation', action='store_true',
                       help='Enable tile rotation (0, 90, 180, 270 degrees) for better matching')
    parser.add_argument('--max-reuse', type=int, default=None,
                       help='Maximum times a tile can be reused (default: unlimited)')
    
    # Blending options
    parser.add_argument('--blend',
                       choices=['none', 'overlay', 'multiply', 'screen', 'soft_light'],
                       default='none',
                       help='Blending mode with original image')
    parser.add_argument('--blend-opacity', type=float, default=0.5,
                       help='Opacity for blending (0.0-1.0, default: 0.5)')
    
    # Enhancement options
    parser.add_argument('--brightness', type=float, default=1.0,
                       help='Brightness multiplier for tiles (default: 1.0)')
    parser.add_argument('--contrast', type=float, default=1.0,
                       help='Contrast multiplier for tiles (default: 1.0)')
    parser.add_argument('--saturation', type=float, default=1.0,
                       help='Saturation multiplier for tiles (default: 1.0)')
    
    # Border options
    parser.add_argument('--border', type=int, default=0,
                       help='Width of border between tiles in pixels (default: 0)')
    parser.add_argument('--border-color', type=int, nargs=3, default=[0, 0, 0],
                       metavar=('R', 'G', 'B'),
                       help='RGB color of borders (default: 0 0 0 = black)')
    
    # Other options
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics after completion')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process mode: process all images in target_image folder')
    
    args = parser.parse_args()
    
    # Validate batch mode
    if args.batch:
        if not args.target_image or not args.tile_folder or not args.output:
            parser.error("Batch mode requires: input_folder tile_folder output_folder")
        if not os.path.exists(args.target_image):
            parser.error(f"Input folder not found: {args.target_image}")
    else:
        # Single image mode
        if not args.target_image or not args.tile_folder or not args.output:
            parser.error("Single image mode requires: target_image tile_folder output")
        if not os.path.exists(args.target_image):
            print(f"Error: Target image not found: {args.target_image}")
            sys.exit(1)
    
    # Prepare arguments
    grid_size = tuple(args.grid) if args.grid else None
    tile_size = tuple(args.tile_size) if args.tile_size else None
    
    color_method_map = {
        'rgb': ColorMatchMethod.RGB_EUCLIDEAN,
        'lab': ColorMatchMethod.LAB_PERCEPTUAL,
        'weighted': ColorMatchMethod.WEIGHTED_RGB
    }
    color_method = color_method_map[args.color_method]
    
    blend_mode_map = {
        'none': BlendMode.NONE,
        'overlay': BlendMode.OVERLAY,
        'multiply': BlendMode.MULTIPLY,
        'screen': BlendMode.SCREEN,
        'soft_light': BlendMode.SOFT_LIGHT
    }
    blend_mode = blend_mode_map[args.blend]
    
    border_color = tuple(args.border_color)
    
    mosaic_kwargs = {
        'grid_size': grid_size,
        'tile_size': tile_size,
        'color_method': color_method,
        'enable_rotation': args.rotation,
        'max_tile_reuse': args.max_reuse,
        'blend_mode': blend_mode,
        'blend_opacity': args.blend_opacity,
        'brightness': args.brightness,
        'contrast': args.contrast,
        'saturation': args.saturation,
        'border_width': args.border,
        'border_color': border_color,
        'sampling_method': args.sampling,
        'show_stats': args.stats
    }
    
    try:
        if args.batch:
            batch_process(
                input_folder=args.target_image,
                tile_folder=args.tile_folder,
                output_folder=args.output,
                **mosaic_kwargs
            )
        else:
            create_mosaic(
                target_image_path=args.target_image,
                tile_folder=args.tile_folder,
                output_path=args.output,
                **mosaic_kwargs
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

