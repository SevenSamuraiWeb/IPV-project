import streamlit as st
import cv2
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torch
from PIL import Image
from model import SaliencyModel
from utils import apply_grabcut, segment_otsu, segment_watershed_sk, segment_canny, segment_kmeans,convert_to_download,blend_background

import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"


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


# -------------------- IMAGE SEGMENTATION FUNCTIONS --------------------
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

# -------------------- SIDEBAR UI --------------------

st.sidebar.title("🧠 Choose Technique:")
mode = st.sidebar.radio("Mode", ["Image segmentation", "Deep Learning"])

if mode == "Image segmentation":
    st.sidebar.markdown("### Segmentation Method")
    method = st.sidebar.selectbox("Select method", ["GrabCut","K-means","Otsu Threshold", "Watershed", "Canny Edges" ])
    if method == "K-means":
        k_val = st.sidebar.radio("K value",[2,3])
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
    st.image(img, caption="📷 Original Image", use_container_width=False)

    with st.spinner("Processing..."):
        if mode == "Image segmentation":
            if method == "GrabCut":
                mask,result = apply_grabcut(img)
                mask = mask.astype(np.uint8) * 255  

            elif method == "Otsu Threshold":
                mask = segment_otsu(img)
                result = blend_background(img, mask)

            elif method == "Watershed":
                # scikit-image watershed
                mask = segment_watershed_sk(img)
                result = blend_background(img, mask)
                            
            elif method == "Canny Edges":
                mask = segment_canny(img)
                result = blend_background(img, mask)
            
            elif method == "K-means":
                mask, segmented = segment_kmeans(img,k=k_val)
                st.image(segmented, caption=f"✅ Segmented image", use_container_width=True)
                result = blend_background(img, mask)
            
            
            col1,col2 = st.columns(2)
            # display mask
            with col1:
                st.image(mask, caption=f"✅ Mask using {method}", use_container_width=True)
                st.download_button("📥 Download", convert_to_download(mask), "mask.png")
            
            # display blended image
            with col2:    
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
