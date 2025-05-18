import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import streamlit as st
from PIL import Image
import io

def blend_background(image, mask, blur_radius=21):
    if mask is None or image is None: 
        return image 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Ensure background is 3-channel
    
    # Ensure mask is 2D
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask_2d = mask[:,:,0]
    elif mask.ndim == 2:
        mask_2d = mask
    else:
        print("Blend_background: Mask has unexpected dimensions.")
        return image 

    mask_float = mask_2d.astype(np.float32) / 255.0
    
    # Ensure blur_radius is odd
    if blur_radius % 2 == 0:
        blur_radius += 1
        
    alpha = cv2.GaussianBlur(mask_float, (blur_radius, blur_radius), 0)
    alpha = np.clip(alpha, 0, 1)[..., np.newaxis] # Add channel dimension for broadcasting
    
    blended = image.astype(np.float32) * alpha + gray_color.astype(np.float32) * (1 - alpha)
    return np.uint8(blended)


def refine_mask(mask, open_iterations=1, close_iterations=1, kernel_size=5):
    """Refines a binary mask using morphological opening and closing."""
    if mask is None or mask.size == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if close_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    if open_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
    return mask


def apply_grabcut(cv_image, iterations=5):
    mask = np.zeros(cv_image.shape[:2], np.uint8)
    
    margin_h = int(cv_image.shape[0] * 0.05)
    margin_w = int(cv_image.shape[1] * 0.05)
    
    rect_x = margin_w
    rect_y = margin_h
    rect_w = cv_image.shape[1] - 2 * margin_w
    rect_h = cv_image.shape[0] - 2 * margin_h

    if rect_w <= 0 or rect_h <= 0: 
        rect_x = 10
        rect_y = 10
        rect_w = cv_image.shape[1] - 20
        rect_h = cv_image.shape[0] - 20
        if rect_w <=0 : rect_w = 1
        if rect_h <=0 : rect_h = 1


    rect = (rect_x, rect_y, rect_w, rect_h)
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(cv_image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
    except Exception as e:
        print(f"GrabCut failed: {e}")

    # mask values: 0 (def BG), 1 (def FG), 2 (prob BG), 3 (prob FG)
    output_mask = np.where((mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 1, 0).astype(np.uint8)
    # output_mask = refine_mask(output_mask, open_iterations=1, close_iterations=1)
    
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Convert single channel gray to 3-channel
    
    # Ensure output_mask is broadcastable (H, W, 1)
    output_mask_3ch = output_mask[:, :, np.newaxis]
    
    blended = gray_background * (1 - output_mask_3ch) + cv_image * output_mask_3ch
    return output_mask,blended.astype(np.uint8)


def convert_to_download(img_array):
    im_pil = Image.fromarray(img_array)
    buffer = io.BytesIO()
    im_pil.save(buffer, format="PNG")
    return buffer.getvalue()


def segment_canny(image, low_thresh_factor=0.66, high_thresh_factor=1.33): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0) 

    # Adaptive Canny thresholds
    median_val = np.median(blur)
    # low_thresh = int(max(0, (1.0 - low_thresh_factor) * median_val if low_thresh_factor < 1 
    #                     else low_thresh_factor * median_val * 0.66)) # ensure low is lower
    # high_thresh = int(min(255, (1.0 + high_thresh_factor) * median_val if high_thresh_factor > 1
    #                      else high_thresh_factor * median_val * 1.33)) # ensure high is higher
    # A simpler common approach:
    sigma = 0.33
    low_thresh = int(max(0, (1.0 - sigma) * median_val))
    high_thresh = int(min(255, (1.0 + sigma) * median_val))


    edges = cv2.Canny(blur, low_thresh, high_thresh)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)) # Slightly larger kernel for closing might be better
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2) # More iterations for closing
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # Refine the mask to fill any remaining small holes or remove small noisy regions
    mask = refine_mask(mask, open_iterations=1, close_iterations=1) 
    return mask


def segment_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute Otsu's threshold. Pixels > ret will be 255 (maxval).
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask.astype(np.uint8)

    # Heuristic: If the "foreground" (white area) is too large, invert the mask.
    # This assumes the primary object of interest isn't the vast majority of the image.
    if np.mean(mask / 255.0) > 0.75: # If > 75% of image is foreground
        mask = cv2.bitwise_not(mask)

    # Refine the mask
    mask = refine_mask(mask, open_iterations=1, close_iterations=2)
    
    fig, ax = plt.subplots(figsize=(3,2))
    ax.hist(gray.ravel(), bins=256, range=(0,256), color='gray', alpha=0.7)
    ax.axvline(ret, color='red', linestyle='--', label=f'Threshold={ret:.0f}')
    ax.legend()
    ax.set_title("Otsu Histogram")
    st.pyplot(fig,use_container_width=True) 
    return mask



def segment_kmeans(image, k=3):
    pixel_vals = image.reshape((-1, 3)).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    segmented_image = centers[labels].reshape(image.shape) # The image quantized by cluster centers

    fg_cluster = 0 # Default
    if k > 0 and labels.size > 0:
        counts = np.bincount(labels, minlength=k) # Ensure counts has k elements
        sorted_indices_by_count = np.argsort(counts) # Indices of clusters sorted by size (ascending)

        if k == 1:
            fg_cluster = 0
        elif k == 2:
            # Assume smaller cluster is foreground
            fg_cluster = sorted_indices_by_count[0]
        else: # k >= 3
            # Assume largest is background (sorted_indices_by_count[k-1])
            # Assume second largest is primary foreground (sorted_indices_by_count[k-2])
            # This is a heuristic. Consider alternative: smallest (sorted_indices_by_count[0])
            # or middle one for k=3 (sorted_indices_by_count[1])
            if k - 2 < len(sorted_indices_by_count):
                 fg_cluster = sorted_indices_by_count[k-2] 
            elif len(sorted_indices_by_count) > 0: # Fallback to smallest if second largest index is out of bounds
                 fg_cluster = sorted_indices_by_count[0]
            # else fg_cluster remains 0 (initial default)

    mask = (labels == fg_cluster).reshape(image.shape[:2]).astype(np.uint8) * 255
    mask = refine_mask(mask, open_iterations=1, close_iterations=1)
    
    return mask, segmented_image 



def segment_watershed_sk(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding - corrected maxval
    # The choice of THRESH_BINARY vs THRESH_BINARY_INV depends on whether objects are lighter or darker than background
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure binary mask (0 or 1) for skimage
    # Clean up noise from Otsu's output
    kernel_morph = np.ones((5,5),np.uint8)
    binary_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_morph, iterations=1) # Remove small noise
    binary_closed = cv2.morphologyEx(binary_opened, cv2.MORPH_CLOSE, kernel_morph, iterations=1) # Fill small holes

    # Compute distance transform on the cleaned binary image
    # Skimage watershed typically works by flooding basins in the NEGATIVE distance transform
    distance = ndi.distance_transform_edt(binary_closed)

    # Get coordinates of local maxima for markers
    # min_distance controls how close markers can be (helps prevent over-segmentation)
    local_maxi = peak_local_max(distance, min_distance=20, labels=binary_closed)
    
    markers = np.zeros(distance.shape, dtype=bool)
    markers[tuple(local_maxi.T)] = True
    markers, _ = ndi.label(markers) # Label the markers

    # Apply watershed. The mask=binary_closed ensures watershed doesn't go outside the Otsu-defined region.
    labels_ws = watershed(-distance, markers, mask=binary_closed)

    # Create final binary mask
    # All labeled regions (except background which might be 0 if not part of initial binary_closed)
    # If labels_ws can contain 0 for background within the mask, then >0 is correct.
    # Typically, watershed labels distinct regions with integers > 0.
    mask = np.uint8(labels_ws > 0) * 255
    mask = refine_mask(mask, open_iterations=1, close_iterations=1)
    return mask