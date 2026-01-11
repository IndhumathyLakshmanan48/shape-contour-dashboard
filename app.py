import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    layout="wide"
)

# ================= HEADER =================
st.markdown(
    "<h1 style='text-align:center;'>Shape & Contour Analyzer Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Interactive Computer Vision System for Geometric Shape Analysis</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("Controls")

uploaded = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

show_contours = st.sidebar.checkbox("Show Contours", True)
show_bbox = st.sidebar.checkbox("Show Bounding Box", True)
show_centroid = st.sidebar.checkbox("Show Centroid", True)
scale = st.sidebar.slider("Pixel to cm Scale", 0.01, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.caption("Contour-based shape detection and feature extraction")

# ================= SHAPE CLASSIFICATION =================
def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    v = len(approx)

    if v == 3:
        return "Triangle"
    elif v == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return "Square" if 0.95 < w / h < 1.05 else "Rectangle"
    elif v == 5:
        return "Pentagon"
    elif v > 6:
        return "Circle"
    else:
        return "Irregular"

# ================= MAIN =================
if uploaded is not None:

    image = np.array(Image.open(uploaded).convert("RGB"))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    display = img.copy()
    results = []

    for i, c in enumerate(contours, 1):
        area_px = cv2.contourArea(c)
        if area_px < 500:
            continue

        perimeter_px = cv2.arcLength(c, True)
        shape = classify_shape(c)

        area_cm = area_px * (scale ** 2)
        perimeter_cm = perimeter_px * scale

        M = cv2.moments(c)
        cx, cy = 0, 0
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        results.append([
            i, shape, round(area_cm, 2), round(perimeter_cm, 2)
        ])

        if show_contours:
            cv2.drawContours(display, [c], -1, (0, 255, 0), 2)

        if show_bbox:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if show_centroid:
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            display, shape, (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )

    # ================= LAYOUT =================
    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.subheader("Processed Image with Detected Shapes")
        st.image(
            cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

    with col2:
        st.subheader("Summary Metrics")
        st.metric("Total Objects Detected", len(results))

    st.markdown("---")

    # ================= TABLE =================
    st.subheader("Geometric Feature Measurements")

    if results:
        df = pd.DataFrame(
            results,
            columns=["Object ID", "Shape Type", "Area (cm²)", "Perimeter (cm)"]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No valid shapes detected.")

    # ================= ANALYTICS =================
    st.subheader("Shape Distribution Analysis")

    if results:
        shape_counts = df["Shape Type"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(shape_counts.index, shape_counts.values)
        ax.set_xlabel("Shape Type")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Detected Geometric Shapes")
        st.pyplot(fig)

else:
    st.info("Upload an image from the left panel to begin analysis.")
