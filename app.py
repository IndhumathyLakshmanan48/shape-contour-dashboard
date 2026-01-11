import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Shape & Contour Analyzer",
    layout="wide"
)

# ================= HEADER =================
st.markdown(
    "<h1 style='text-align:center;'>🧠 Smart Shape & Contour Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Interactive Computer Vision Dashboard</p>",
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.header("⚙ Controls")

uploaded = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

show_contours = st.sidebar.checkbox("Show Contours", True)
show_bbox = st.sidebar.checkbox("Show Bounding Box", True)
show_centroid = st.sidebar.checkbox("Show Centroid", True)
scale = st.sidebar.slider("Pixel → cm Scale", 0.01, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.info("Contour-based shape detection and analysis")

# ================= SHAPE CLASSIFIER =================
def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return "Square" if 0.95 < w / h < 1.05 else "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices > 6:
        return "Circle"
    else:
        return "Irregular"

# ================= MAIN DASHBOARD =================
if uploaded is not None:

    # Load image
    image = np.array(Image.open(uploaded).convert("RGB"))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    # Contour detection
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    display = img.copy()
    shapes = []
    areas_cm = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        shape = classify_shape(c)
        shapes.append(shape)
        areas_cm.append(area * (scale ** 2))

        if show_contours:
            cv2.drawContours(display, [c], -1, (0, 255, 0), 2)

        if show_bbox:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if show_centroid:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

    # ================= LAYOUT =================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📷 Processed Image")
        st.image(
            cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

    with col2:
        st.subheader("📊 Metrics")
        st.metric("Total Objects", len(shapes))
        st.write("Detected Shapes:")
        st.write(shapes)

    # ================= ANALYTICS =================
    st.subheader("📈 Shape Analytics")

    if len(shapes) > 0:
        unique_shapes = list(dict.fromkeys(shapes))
        counts = [shapes.count(s) for s in unique_shapes]

        fig, ax = plt.subplots()
        ax.bar(unique_shapes, counts)
        ax.set_xlabel("Shape Type")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Detected Shapes")
        st.pyplot(fig)
    else:
        st.info("No shapes detected for analysis.")

else:
    st.warning("⬅ Please upload an image from the sidebar to start analysis.")
