import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# ==============================
# Utility Functions
# ==============================
def plot_histogram(img, title="Histogram"):
    """Return a matplotlib figure of histogram."""
    fig, ax = plt.subplots()
    if len(img.shape) == 2:  # Grayscale
        ax.hist(img.ravel(), bins=256, range=(0, 256), color="black")
    else:  # Color
        colors = ("b", "g", "r")
        for i, c in enumerate(colors):
            ax.hist(img[:, :, i].ravel(), bins=256, range=(0, 256), color=c, alpha=0.6)
    ax.set_title(title)
    ax.set_xlim([0, 256])
    return fig


def linear_negative(img):
    return 255 - img


def contrast_stretching(img):
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


def piecewise_linear(img, r1, s1, r2, s2):
    # Simple piecewise linear mapping
    xp = [0, r1, r2, 255]
    fp = [0, s1, s2, 255]
    table = np.interp(np.arange(256), xp, fp).astype("uint8")
    return cv2.LUT(img, table)


def log_transform(img, c=1.0):
    img_float = img.astype(np.float32) / 255.0
    log_img = c * np.log1p(img_float)
    log_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(log_img)


def gamma_transform(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)


def hist_equalization(img):
    if len(img.shape) == 2:  # Grayscale
        return cv2.equalizeHist(img)
    else:  # Color → apply to Y channel in YCrCb
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def adaptive_hist_equalization(img, clip_limit=0.01):
    return exposure.equalize_adapthist(img, clip_limit=clip_limit) * 255


def clahe_equalization(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ==============================
# Streamlit App
# ==============================
st.title("📷 Interactive Image Processing Toolkit")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Option to convert to grayscale
    mode = st.radio("Image Mode", ["Color", "Grayscale"])
    if mode == "Grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.subheader("Original Image & Histogram")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, channels="BGR" if mode=="Color" else "GRAY", use_container_width=True)
    with col2:
        st.pyplot(plot_histogram(img, "Original Histogram"))

    # Processing method
    method = st.selectbox(
        "Select Processing Method",
        [
            "Linear Negative",
            "Contrast Stretching",
            "Piecewise Linear Transformation",
            "Log Transformation",
            "Gamma Transformation",
            "Histogram Equalization",
            "Adaptive Histogram Equalization",
            "CLAHE"
        ]
    )

    # Parameters
    processed = None
    if method == "Linear Negative":
        processed = linear_negative(img)

    elif method == "Contrast Stretching":
        processed = contrast_stretching(img)

    elif method == "Piecewise Linear Transformation":
        r1 = st.slider("r1", 0, 255, 70)
        s1 = st.slider("s1", 0, 255, 0)
        r2 = st.slider("r2", 0, 255, 140)
        s2 = st.slider("s2", 0, 255, 255)
        processed = piecewise_linear(img, r1, s1, r2, s2)

    elif method == "Log Transformation":
        c = st.slider("Scaling Constant (c)", 1, 10, 5)
        processed = log_transform(img, c=c)

    elif method == "Gamma Transformation":
        gamma = st.slider("Gamma", 0.1, 5.0, 1.0, step=0.1)
        processed = gamma_transform(img, gamma)

    elif method == "Histogram Equalization":
        processed = hist_equalization(img)

    elif method == "Adaptive Histogram Equalization":
        clip_limit = st.slider("Clip Limit", 0.001, 0.1, 0.01)
        processed = adaptive_hist_equalization(img, clip_limit=clip_limit).astype("uint8")

    elif method == "CLAHE":
        clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0)
        tile_size = st.slider("Tile Grid Size", 2, 16, 8)
        processed = clahe_equalization(img, clip_limit=clip_limit, tile_grid_size=(tile_size, tile_size))

    # Show results
    st.subheader("Processed Image & Histogram")
    col1, col2 = st.columns(2)
    with col1:
        st.image(processed, channels="BGR" if mode=="Color" else "GRAY", use_container_width=True)
    with col2:
        st.pyplot(plot_histogram(processed, "Processed Histogram"))
