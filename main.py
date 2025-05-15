import streamlit as st
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Subject Highlighter", layout="wide")
st.title("🖼️ Subject Highlighter & Background Modifier")
st.markdown("Upload an image to get:")
st.markdown("- ✅ **Highlighted Subjects**")
st.markdown("- 🌫️ **Blurred Background** version")
st.markdown("- 🩶 **Grey Background** with colored subjects")
st.markdown("- 📥 Download options")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Load Mask R-CNN
@st.cache_resource
def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

model = load_model()

def process_image(pil_image, threshold=0.75):
    image_np = np.array(pil_image.convert("RGB"))
    input_tensor = F.to_tensor(image_np)

    with torch.no_grad():
        predictions = model([input_tensor])[0]

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

# Display
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("Processing..."):
        contoured, blurred_bg, greyed_bg = process_image(image)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(contoured, caption="🔵 Highlighted Subjects", use_column_width=True)
            st.download_button("📥 Download", convert_to_download(contoured), "highlighted.png")

        with col2:
            st.image(blurred_bg, caption="🌫️ Blurred Background", use_column_width=True)
            st.download_button("📥 Download", convert_to_download(blurred_bg), "blurred_background.png")

        with col3:
            st.image(greyed_bg, caption="🩶 Grey Background", use_column_width=True)
            st.download_button("📥 Download", convert_to_download(greyed_bg), "grey_background.png")
