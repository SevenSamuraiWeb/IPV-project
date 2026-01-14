# Deep Learning Image Segmentation Tool

This project represents a comprehensive solution for salient object detection and background manipulation. It leverages both state-of-the-art Deep Learning models and classical Computer Vision techniques to provide a versatile tool for subject highlighting and background replacement.

## 🚀 Features

The application provides a user-friendly interface powered by **Streamlit**, allowing users to upload images and apply various segmentation techniques.

### 🧠 Deep Learning Methods
*   **U-Net with VGG16 Backbone**: A custom-trained U-Net model designed for high-precision salient object detection. The encoder uses a pre-trained VGG16 (up to layer `conv4_3`) to extract robust features, while the decoder reconstructs the segmentation mask. This model is particularly effective on datasets like DUTS.
*   **Mask R-CNN**: Uses a pre-trained ResNet50-FPN model from `torchvision` for instance segmentation. It can detect multiple objects and provide high-quality masks.

### 📷 Classical Computer Vision Techniques
For comparison and specific use-cases, the tool also includes:
*   **GrabCut**: Interactive foreground extraction using iterated graph cuts.
*   **Otsu Thresholding**: Automatic global thresholding based on histogram analysis.
*   **Watershed (scikit-image)**: Marker-based segmentation treating the image pixel gradient as a topographic surface.
*   **Canny Edge Detection**: Edge-based segmentation combined with morphological operations.
*   **K-Means Clustering**: Unsupervised segmentation by clustering pixel colors.

### 🎨 Image Processing capabilities
*   **Subject Highlighting**: Draws contours or overlays masks on the detected subject.
*   **Background Blurring (Bokeh Effect)**: Automatically blurs the background while keeping the subject sharp.
*   **Background Desaturation**: Converts the background to grayscale/black-and-white while preserving the subject's original colors.
*   **Export**: Results can be downloaded directly from the interface.

## 📂 Project Structure

*   `main.py`: The entry point for the Streamlit application. Handles UI, model loading, and interaction logic.
*   `model.py`: Defines the `SaliencyModel`, `SimpleEncoder` (VGG16), and `SimpleDecoder` architecture.
*   `utils.py`: Contains helper functions for image processing (`blend_background`, `refine_mask`) and implementations of classical segmentation algorithms (`apply_grabcut`, `segment_otsu`, etc.).
*   `unet.pth`: The trained weights for the custom U-Net model from the DUTS dataset.
*   `requirements.txt`: List of Python dependencies.

## 🛠️ Installation & Setup

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install Dependencies**:
    Ensure you have Python installed (recommended version 3.8+). Run the following command:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    Start the Streamlit server:
    ```bash
    streamlit run main.py
    ```

4.  **Access the Tool**:
    The application will automatically open in your default web browser at `http://localhost:8501`.

## 🖥️ Usage

1.  **Upload Image**: Use the file uploader to select an image (`.jpg`, `.png`, `.jpeg`, `.bmp`).
2.  **Select Mode**: Choose between "Image segmentation" (Classical) and "Deep Learning" in the sidebar.
3.  **Choose Method**: Select the specific algorithm or model you want to use.
4.  **View & Download**: The processed images (Mask, Highlighted, Blurred Background, Grey Background) will be displayed. Click the "Download" button to save them.

## 🔍 Model Details

The **SaliencyModel** is a U-Net architecture:
*   **Encoder**: VGG16 with Batch Normalization (pretrained on ImageNet). Layers are used up to `features[22]`.
*   **Decoder**: A series of `ConvTranspose2d` layers with ReLU activation, progressively upsampling the feature map to the original resolution. A final `Sigmoid` activation produces the probability map.

## ⚠️ Notes
*   The application uses a GPU if available (`cuda`), otherwise falls back to CPU.
*   The `unet.pth` file must be present in the root directory for the U-Net model to work.
