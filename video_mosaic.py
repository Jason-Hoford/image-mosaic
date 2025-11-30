#!/usr/bin/env python3
"""
Video Frame Mosaic Creator

Extracts frames from video files at 1-second intervals and uses them as tiles
to recreate any target image as a mosaic. Perfect for creating artistic mosaics
from movie scenes, video clips, or any video content.
"""

import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Please run: pip install opencv-python")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is not installed. Please run: pip install Pillow")
    sys.exit(1)

# Import mosaic creation functions from image_mosaic.py
try:
    from image_mosaic import (
        ColorMatchMethod, BlendMode, create_mosaic,
        load_tile_images, get_average_color
    )
except ImportError:
    print("Error: Could not import from image_mosaic.py. Ensure image_mosaic.py is in the same directory.")
    sys.exit(1)


def get_video_info(video_path: str) -> Tuple[float, int, int, int, float]:
    """
    Get video information including FPS, frame count, width, height, and duration.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (fps, frame_count, width, height, duration)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return fps, frame_count, width, height, duration


def extract_frames_from_video(video_path: str,
                              output_folder: str,
                              interval_seconds: float = 1.0,
                              max_frames: Optional[int] = None,
                              resize: Optional[Tuple[int, int]] = None) -> int:
    """
    Extract frames from a video at specified time intervals.
    
    Args:
        video_path: Path to input video file
        output_folder: Folder to save extracted frames
        interval_seconds: Time interval between frames in seconds (default: 1.0)
        max_frames: Maximum number of frames to extract (None = all)
        resize: Optional (width, height) tuple to resize frames
        
    Returns:
        Number of frames extracted
    """
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video info: {width}x{height}, {fps:.2f} FPS, {duration:.2f} seconds, {frame_count} frames")
    
    if fps == 0:
        raise ValueError("Could not determine video FPS")
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate frame interval
    frame_interval = int(fps * interval_seconds)
    if frame_interval < 1:
        frame_interval = 1
    
    # Calculate expected number of frames
    expected_frames = int(duration / interval_seconds) + 1
    if max_frames:
        expected_frames = min(expected_frames, max_frames)
    
    print(f"Extracting frames every {interval_seconds} seconds (every {frame_interval} frames)")
    print(f"Expected to extract approximately {expected_frames} frames")
    
    extracted_count = 0
    current_frame = 0
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame if it matches the interval
        if current_frame % frame_interval == 0:
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if specified
            if resize:
                frame_rgb = cv2.resize(frame_rgb, resize, interpolation=cv2.INTER_LANCZOS4)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Save frame
            timestamp = current_frame / fps
            frame_filename = f"frame_{frame_number:06d}_t{timestamp:.2f}s.jpg"
            frame_path = output_path / frame_filename
            pil_image.save(frame_path, quality=95)
            
            extracted_count += 1
            frame_number += 1
            
            if extracted_count % 10 == 0:
                print(f"Extracted {extracted_count} frames...", end='\r')
            
            # Check max frames limit
            if max_frames and extracted_count >= max_frames:
                break
        
        current_frame += 1
    
    cap.release()
    print(f"\nSuccessfully extracted {extracted_count} frames to {output_folder}")
    
    return extracted_count


def create_video_mosaic(video_path: str,
                        target_image_path: str,
                        output_path: str,
                        interval_seconds: float = 1.0,
                        max_frames: Optional[int] = None,
                        frame_resize: Optional[Tuple[int, int]] = None,
                        keep_frames: bool = False,
                        frames_folder: Optional[str] = None,
                        **mosaic_kwargs) -> None:
    """
    Create a mosaic from video frames.
    
    Args:
        video_path: Path to input video file
        target_image_path: Path to target image to recreate
        output_path: Path to save the output mosaic
        interval_seconds: Time interval between frames in seconds (default: 1.0)
        max_frames: Maximum number of frames to extract (None = all)
        frame_resize: Optional (width, height) to resize frames before using as tiles
        keep_frames: If True, keep extracted frames after processing
        frames_folder: Custom folder for frames (None = temporary folder)
        **mosaic_kwargs: Additional arguments passed to create_mosaic
    """
    # Validate video file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(target_image_path):
        raise FileNotFoundError(f"Target image not found: {target_image_path}")
    
    # Get video info
    try:
        fps, frame_count, width, height, duration = get_video_info(video_path)
        print(f"\nVideo: {os.path.basename(video_path)}")
        print(f"Duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Error reading video: {e}")
        sys.exit(1)
    
    # Setup frames folder
    use_temp = frames_folder is None
    if use_temp:
        frames_folder = tempfile.mkdtemp(prefix="video_frames_")
        print(f"Using temporary folder for frames: {frames_folder}")
    else:
        frames_folder_path = Path(frames_folder)
        frames_folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Using frames folder: {frames_folder}")
    
    try:
        # Extract frames
        print("\n" + "="*60)
        print("STEP 1: Extracting frames from video")
        print("="*60)
        extracted_count = extract_frames_from_video(
            video_path=video_path,
            output_folder=frames_folder,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
            resize=frame_resize
        )
        
        if extracted_count == 0:
            raise ValueError("No frames were extracted from the video")
        
        # Create mosaic
        print("\n" + "="*60)
        print("STEP 2: Creating mosaic from extracted frames")
        print("="*60)
        create_mosaic(
            target_image_path=target_image_path,
            tile_folder=frames_folder,
            output_path=output_path,
            **mosaic_kwargs
        )
        
        print("\n" + "="*60)
        print("SUCCESS!")
        print("="*60)
        print(f"Mosaic created: {output_path}")
        print(f"Frames extracted: {extracted_count}")
        if not keep_frames and use_temp:
            print(f"Frames folder (temporary): {frames_folder}")
        elif keep_frames:
            print(f"Frames saved to: {frames_folder}")
        
    finally:
        # Cleanup temporary folder if not keeping frames
        if use_temp and not keep_frames:
            print(f"\nCleaning up temporary frames folder...")
            shutil.rmtree(frames_folder, ignore_errors=True)
            print("Cleanup complete")


def main():
    """Main entry point for the video mosaic tool."""
    parser = argparse.ArgumentParser(
        description='Create image mosaics using frames extracted from video files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - extract 1 frame per second
  python video_mosaic.py video.mp4 target.jpg output.jpg
  
  # Extract frames every 0.5 seconds with high quality settings
  python video_mosaic.py video.mp4 target.jpg output.jpg --interval 0.5 --color-method lab --rotation
  
  # Limit to 500 frames and keep extracted frames
  python video_mosaic.py video.mp4 target.jpg output.jpg --max-frames 500 --keep-frames
  
  # Resize frames before using as tiles
  python video_mosaic.py video.mp4 target.jpg output.jpg --frame-size 200 200
  
  # Full featured example
  python video_mosaic.py movie.mp4 photo.jpg mosaic.jpg --interval 1.0 --grid 100 100 --color-method lab --rotation --blend overlay --stats
        """
    )
    
    # Required arguments
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('target_image', help='Path to target image to recreate as mosaic')
    parser.add_argument('output', help='Path to save the output mosaic image')
    
    # Frame extraction options
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Time interval between frames in seconds (default: 1.0)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to extract (default: all)')
    parser.add_argument('--frame-size', type=int, nargs=2, default=None,
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Resize extracted frames to this size before using as tiles')
    parser.add_argument('--keep-frames', action='store_true',
                       help='Keep extracted frames after processing (default: delete temporary frames)')
    parser.add_argument('--frames-folder', type=str, default=None,
                       help='Custom folder to save extracted frames (default: temporary folder)')
    
    # Mosaic options (from main.py)
    parser.add_argument('--grid', type=int, nargs=2, metavar=('ROWS', 'COLS'),
                       help='Grid size (rows cols). If not specified, calculated automatically')
    parser.add_argument('--tile-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       dest='tile_size',
                       help='Size of each tile in pixels (width height)')
    parser.add_argument('--color-method', choices=['rgb', 'lab', 'weighted'],
                       default='rgb', help='Color matching algorithm')
    parser.add_argument('--sampling', choices=['average', 'center', 'dominant'],
                       default='average', help='Method for sampling region colors')
    parser.add_argument('--rotation', action='store_true',
                       help='Enable tile rotation for better matching')
    parser.add_argument('--max-reuse', type=int, default=None,
                       help='Maximum times a tile can be reused')
    parser.add_argument('--blend', choices=['none', 'overlay', 'multiply', 'screen', 'soft_light'],
                       default='none', help='Blending mode with original image')
    parser.add_argument('--blend-opacity', type=float, default=0.5,
                       help='Opacity for blending (0.0-1.0)')
    parser.add_argument('--brightness', type=float, default=1.0,
                       help='Brightness multiplier for tiles')
    parser.add_argument('--contrast', type=float, default=1.0,
                       help='Contrast multiplier for tiles')
    parser.add_argument('--saturation', type=float, default=1.0,
                       help='Saturation multiplier for tiles')
    parser.add_argument('--border', type=int, default=0,
                       help='Width of border between tiles in pixels')
    parser.add_argument('--border-color', type=int, nargs=3, default=[0, 0, 0],
                       metavar=('R', 'G', 'B'), help='RGB color of borders')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics after completion')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.target_image):
        print(f"Error: Target image not found: {args.target_image}")
        sys.exit(1)
    
    # Prepare mosaic arguments
    grid_size = tuple(args.grid) if args.grid else None
    tile_size = tuple(args.tile_size) if args.tile_size else None
    frame_resize = tuple(args.frame_size) if args.frame_size else None
    
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
        create_video_mosaic(
            video_path=args.video,
            target_image_path=args.target_image,
            output_path=args.output,
            interval_seconds=args.interval,
            max_frames=args.max_frames,
            frame_resize=frame_resize,
            keep_frames=args.keep_frames,
            frames_folder=args.frames_folder,
            **mosaic_kwargs
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
