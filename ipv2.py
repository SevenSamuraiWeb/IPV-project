import streamlit as st
import cv2
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torch
from PIL import Image
import io
from unet import SaliencyModel

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Subject Highlighter", layout="wide")

# -------------------- MODEL LOADING --------------------

# mask R CNN
@st.cache_resource
def load_mask_r_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

mask_r_model = load_mask_r_model()

# U-Net with VGG16 encoder
@st.cache_resource
@st.cache_resource
def load_unet_model(pth_path, device):
    model = SaliencyModel()
    state_dict = torch.load(pth_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet_model = load_unet_model('unet.pth',device)


# -------------------- UTILITY FUNCTIONS --------------------

def segment_otsu(image):
    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute Otsu's threshold
    ret, mask = cv2.threshold(gray, 255, 0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = mask.astype(np.uint8)
    elapsed = time.time() - start
    logging.info(f"Otsu's method: threshold={ret:.2f}, time={elapsed:.3f}s, mask coverage={(np.mean(mask>0)*100):.2f}%")
    
    # Plot histogram and threshold using Streamlit
    fig, ax = plt.subplots(figsize=(3,2))
    ax.hist(gray.ravel(), bins=256, range=(0,256), color='gray', alpha=0.7)
    ax.axvline(ret, color='red', linestyle='--', label=f'Threshold={ret:.0f}')
    ax.legend()
    ax.set_title("Otsu Histogram")
    st.pyplot(fig)
    return mask


def segment_kmeans(image, k=3):
    start = time.time()
    pixel_vals = image.reshape((-1, 3)).astype(np.float32)
    
    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    segmented = centers[labels].reshape(image.shape)
    fg_cluster = 0
    mask = (labels == fg_cluster).reshape(image.shape[:2]).astype(np.uint8) * 255
    elapsed = time.time() - start

    logging.info(f"K-Means (k={k}): time={elapsed:.3f}s, cluster sizes={np.bincount(labels)}, chosen_fg={fg_cluster}")
    return mask, segmented


def segment_watershed(image):
    start = time.time()

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding — fix: parameters reversed
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure binary mask for distance transform
    binary = np.uint8(thresh == 255)

    # Compute the distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Detect local maxima as markers — peak_local_max returns boolean mask
    local_max = peak_local_max(dist, min_distance=20, labels=binary)

    # Label markers
    markers, _ = ndi.label(local_max)

    # Apply watershed
    labels_ws = watershed(-dist, markers, mask=binary)

    # Create output mask
    mask = np.uint8(labels_ws > 0) * 255

    # Logging metrics
    elapsed = time.time() - start
    logging.info(f"Watershed: time={elapsed:.3f}s, markers={markers.max()}, mask coverage={(np.mean(mask > 0) * 100):.2f}%")
    return mask



def segment_canny(image, low_thresh=50, high_thresh=150):
    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1.4)
    edges = cv2.Canny(blur, low_thresh, high_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    elapsed = time.time() - start
    logging.info(f"Canny+Morphology: time={elapsed:.3f}s, edges pixels={(np.mean(edges>0)*100):.2f}%, mask coverage={(np.mean(mask>0)*100):.2f}%")
    return mask


def apply_grabcut(cv_image, iterations=5):
    mask = np.zeros(cv_image.shape[:2], np.uint8)
    rect = (10, 10, cv_image.shape[1] - 20, cv_image.shape[0] - 20)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(cv_image, mask, rect, bgdModel, fgdModel, iterations, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
    
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blended = gray_background * (1 - mask2[:, :, np.newaxis]) + cv_image * (mask2[:, :, np.newaxis])
    return blended.astype(np.uint8)


def blend_background(image, mask, blur_radius=21):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    mask_float = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(mask_float, (blur_radius, blur_radius), 0)
    alpha = np.clip(alpha, 0, 1)[..., np.newaxis]
    
    blended = image.astype(np.float32) * alpha + gray_color.astype(np.float32) * (1 - alpha)
    return np.uint8(blended)

def process_image_unet(pil_image, model, device, threshold=0.5):
    image_np = np.array(pil_image.convert("RGB"))
    h, w = image_np.shape[:2]

    # Resize input for UNet
    image_resized = cv2.resize(image_np, (224, 224))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    input_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        saliency_pred = model(input_tensor)[0, 0].cpu().numpy()

    # Resize saliency back to original size (H, W)
    saliency_map = cv2.resize(saliency_pred, (w, h))

    # Create blurred and grey background versions
    blurred_img = cv2.GaussianBlur(image_np, (21, 21), 30)
    grey_bg_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    grey_bg_img = np.stack([grey_bg_img]*3, axis=-1)

    # Threshold saliency map for contour drawing (optional)
    binary_mask = (saliency_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contoured_img = image_np.copy()
    color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
    cv2.drawContours(contoured_img, contours, -1, color, 2)

    # Use soft mask for blending (scale saliency_map to 3 channels)
    saliency_3ch = np.repeat(saliency_map[:, :, None], 3, axis=2)

    # Blend original with blurred background by saliency map
    combined_blur = (image_np * saliency_3ch + blurred_img * (1 - saliency_3ch)).astype(np.uint8)

    # Blend original with grey background by saliency map
    combined_grey = (image_np * saliency_3ch + grey_bg_img * (1 - saliency_3ch)).astype(np.uint8)

    return contoured_img, combined_blur, combined_grey



def process_image_maskR(pil_image, threshold=0.75):
    image_np = np.array(pil_image.convert("RGB"))
    input_tensor = F.to_tensor(image_np)

    with torch.no_grad():
        predictions = mask_r_model([input_tensor])[0]

    masks = predictions['masks']
    scores = predictions['scores']

    contoured_img = image_np.copy()
    blurred_img = cv2.GaussianBlur(image_np, (21, 21), 30)
    grey_bg_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    grey_bg_img = np.stack([grey_bg_img]*3, axis=-1)  # convert to 3 channels

    combined_blur = blurred_img.copy()
    combined_grey = grey_bg_img.copy()

    for i in range(len(masks)):
        if scores[i] >= threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contoured_img, contours, -1, color.tolist(), 2)

            subject_mask = mask > 127
            combined_blur[subject_mask] = image_np[subject_mask]
            combined_grey[subject_mask] = image_np[subject_mask]

    return contoured_img, combined_blur, combined_grey

def convert_to_download(img_array):
    im_pil = Image.fromarray(img_array)
    buffer = io.BytesIO()
    im_pil.save(buffer, format="PNG")
    return buffer.getvalue()

# -------------------- SIDEBAR UI --------------------

st.sidebar.title("🧠 Choose Technique:")
mode = st.sidebar.radio("Mode", ["Image segmentation", "Deep Learning"])

if mode == "Image segmentation":
    st.sidebar.markdown("### Segmentation Method")
    method = st.sidebar.selectbox("Select method", ["GrabCut","K-means","Otsu Threshold", "Watershed", "Canny Edges" ])
else:
    st.sidebar.markdown("### CNN model")
    method = st.sidebar.selectbox("Select method", ["Mask-R-CNN","U-Net w/ VGG16"])

# -------------------- MAIN INTERFACE --------------------
st.title("🖼️ Subject Highlighter & Background Modifier")
st.markdown("Upload an image to get:")
st.markdown("- ✅ **Highlighted Subjects**")
st.markdown("- 🌫️ **Blurred Background** version")
st.markdown("- 🩶 **Grey Background** with colored subjects")
st.markdown("- 📥 Download options")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img = np.array(img_pil)
    st.image(img, caption="📷 Original Image", use_container_width=True)

    with st.spinner("Processing..."):
        if mode == "Image segmentation":
            if method == "GrabCut":
                mask = apply_grabcut(img)
                result = mask  # already blended    
            elif method == "Otsu Threshold":
                mask = segment_otsu(img)
                result = blend_background(img, mask)
            elif method == "Watershed":
                mask = segment_watershed(img)
                result = blend_background(img, mask)
            elif method == "Canny Edges":
                mask = segment_canny(img)
                result = blend_background(img, mask)
            elif method == "K-means":
                mask, _ = segment_kmeans(img)
                result = blend_background(img, mask)

            st.image(result, caption=f"✅ Result using {method}", use_container_width=True)
            st.download_button("📥 Download", convert_to_download(result), "highlighted.png")

        else:         
                if method == "Mask-R-CNN":
                    contoured, blurred_bg, greyed_bg = process_image_maskR(img_pil)
                elif method == "U-Net w/ VGG16":
                    contoured, blurred_bg, greyed_bg = process_image_unet(img_pil,model=unet_model,device=device) 
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image(contoured, caption="🔵 Highlighted Subjects", use_container_width=True)
                    st.download_button("📥 Download", convert_to_download(contoured), "highlighted.png")

                with col2:
                    st.image(blurred_bg, caption="🌫️ Blurred Background", use_container_width=True)
                    st.download_button("📥 Download", convert_to_download(blurred_bg), "blurred_background.png")

                with col3:
                    st.image(greyed_bg, caption="🩶 Grey Background", use_container_width=True)
                    st.download_button("📥 Download", convert_to_download(greyed_bg), "grey_background.png")
else:
    st.info("👈 Upload an image to get started.")
