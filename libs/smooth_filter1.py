import cv2
import torch
import numpy as np
from PIL import Image
import os

def smooth_filter_enhanced(init_img, content_img, f_radius=15, f_edge=1e-1):
    """
    Enhanced smooth filter with multiple stages to reduce artifacts.
    
    Args:
        init_img: Initial stylized image (PIL Image, numpy array, or path)
        content_img: Content image (PIL Image, numpy array, or path)
        f_radius: Filter radius
        f_edge: Filter edge preservation strength
        
    Returns:
        PIL Image with the smoothed result
    """
    # Load images
    if isinstance(init_img, str):
        init_img = Image.open(init_img).convert("RGB")
    
    if isinstance(content_img, str):
        content_img = Image.open(content_img).convert("RGB")
    
    # Convert to numpy arrays
    if isinstance(init_img, Image.Image):
        stylized_np = np.array(init_img)
    else:
        stylized_np = init_img.copy()  # Assume numpy array
    
    h, w, _ = stylized_np.shape
    
    # Resize content to match stylized
    if isinstance(content_img, Image.Image):
        content_np = np.array(content_img.resize((w, h)))
    elif isinstance(content_img, np.ndarray):
        if content_img.shape[:2] != (h, w):
            content_np = cv2.resize(content_img, (w, h))
        else:
            content_np = content_img.copy()
    
    # Convert to float32 for processing
    stylized_float = stylized_np.astype(np.float32) / 255.0
    content_float = content_np.astype(np.float32) / 255.0
    
    # Convert to BGR for OpenCV processing
    stylized_bgr = cv2.cvtColor(stylized_np, cv2.COLOR_RGB2BGR)
    content_bgr = cv2.cvtColor(content_np, cv2.COLOR_RGB2BGR)
    
    # 1. Apply guided filter first (uses content as guide)
    radius = int(f_radius)
    eps = f_edge * 0.1
    guided_filter_result = cv2.ximgproc.guidedFilter(
        guide=content_bgr,
        src=stylized_bgr,
        radius=radius,
        eps=eps
    )
    
    # 2. Apply domain transform filter (edge-preserving)
    try:
        dt_filter_result = cv2.ximgproc.dtFilter(
            guide=content_bgr,
            src=guided_filter_result,
            sigmaSpatial=f_radius,
            sigmaColor=f_edge,
            mode=cv2.ximgproc.DTF_RF  # Rolling guidance filter mode
        )
    except Exception as e:
        print(f"Domain transform filter failed: {e}")
        # Fall back to recursive edge-preserving filter
        dt_filter_result = cv2.edgePreservingFilter(
            guided_filter_result, 
            flags=cv2.RECURS_FILTER,
            sigma_s=f_radius,
            sigma_r=f_edge
        )
    
    # 3. Apply bilateral filter to reduce remaining noise while preserving edges
    try:
        bilateral_result = cv2.bilateralFilter(
            dt_filter_result,
            d=9,  # Diameter of pixel neighborhood
            sigmaColor=f_edge * 100,
            sigmaSpace=f_radius / 4
        )
    except Exception as e:
        print(f"Bilateral filter failed: {e}")
        bilateral_result = dt_filter_result
    
    # 4. Structure-texture decomposition to preserve important structure
    try:
        # Use relative total variation for structure-texture decomposition
        structure_result = cv2.ximgproc.l0Smooth(
            bilateral_result,
            lambda_=0.02
        )
        
        # Blend structure with filtered result
        alpha = 0.6  # Adjust blend factor
        blended_result = cv2.addWeighted(
            structure_result, alpha,
            bilateral_result, 1 - alpha,
            0
        )
    except Exception as e:
        print(f"Structure-texture decomposition failed: {e}")
        blended_result = bilateral_result
    
    # Convert back to RGB
    filtered_rgb = cv2.cvtColor(blended_result, cv2.COLOR_BGR2RGB)
    
    # Return as PIL image
    return Image.fromarray(filtered_rgb)

def smooth_filter_fallback(init_img, content_img, f_radius=15, f_edge=1e-1):
    """
    Basic fallback smooth filter method.
    """
    # Load images
    if isinstance(init_img, str):
        init_img = Image.open(init_img).convert("RGB")
    
    if isinstance(content_img, str):
        content_img = Image.open(content_img).convert("RGB")
    
    # Convert to numpy arrays
    if isinstance(init_img, Image.Image):
        stylized_np = np.array(init_img)
    else:
        stylized_np = init_img.copy()  # Assume numpy array
    
    h, w, _ = stylized_np.shape
    
    # Resize content to match stylized
    if isinstance(content_img, Image.Image):
        content_np = np.array(content_img.resize((w, h)))
    elif isinstance(content_img, np.ndarray):
        if content_img.shape[:2] != (h, w):
            content_np = cv2.resize(content_img, (w, h))
        else:
            content_np = content_img.copy()
    
    # Convert to BGR for OpenCV
    stylized_bgr = cv2.cvtColor(stylized_np, cv2.COLOR_RGB2BGR)
    content_bgr = cv2.cvtColor(content_np, cv2.COLOR_RGB2BGR)
    
    # Simple bilateral filter
    filtered_bgr = cv2.bilateralFilter(
        stylized_bgr, 
        d=int(f_radius), 
        sigmaColor=f_edge*100, 
        sigmaSpace=f_radius/2
    )
    
    # Convert back to RGB
    filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
    
    # Return as PIL image
    return Image.fromarray(filtered_rgb)

def smooth_filter(init_img, content_img, f_radius=15, f_edge=1e-1):
    """
    Main smooth filter function that tries multiple approaches.
    
    Args:
        init_img: Initial stylized image (PIL Image, numpy array, or path)
        content_img: Content image (PIL Image, numpy array, or path)
        f_radius: Filter radius
        f_edge: Filter edge preservation strength
        
    Returns:
        PIL Image with the smoothed result
    """
    try:
        # Try the enhanced filter first
        print("Applying enhanced filtering...")
        return smooth_filter_enhanced(init_img, content_img, f_radius, f_edge)
    except Exception as e:
        print(f"Enhanced filtering failed: {e}")
        print("Falling back to basic filtering...")
        try:
            # Fall back to simple method
            return smooth_filter_fallback(init_img, content_img, f_radius, f_edge)
        except Exception as e2:
            print(f"Fallback filtering also failed: {e2}")
            # If all else fails, return original image
            if isinstance(init_img, str):
                return Image.open(init_img).convert("RGB")
            elif isinstance(init_img, np.ndarray):
                return Image.fromarray(init_img)
            else:
                return init_img