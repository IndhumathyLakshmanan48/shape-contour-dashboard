import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

# ================= HEADER =================
st.markdown("<h2 style='text-align:center;'>Shape & Contour Analyzer</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Geometric Shape Detection, Feature Extraction & Visual Analysis</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Detection Mode",
    ["Shape Mode", "Document Mode"],
    help=(
        "Shape Mode: Detects multiple geometric objects.\n\n"
        "Document Mode: Detects a single dominant document-like object."
    )
)

uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

show_contours = st.sidebar.checkbox("Show Contours", True)
show_bbox = st.sidebar.checkbox("Show Bounding Box", True)
show_centroid = st.sidebar.checkbox("Show Centroid", True)

scale = st.sidebar.slider("Pixel → cm Scale", 0.01, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Legend")
st.sidebar.markdown(
    """
    <div style="font-size:14px">
    <span style="color:#00FF00;">■</span> Contours<br>
    <span style="color:#0000FF;">■</span> Bounding Box<br>
    <span style="color:#FF0000;">■</span> Centroid
    </div>
    """,
    unsafe_allow_html=True
)

# ================= SHAPE FUNCTIONS =================
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

def shape_complexity(vertices):
    if vertices <= 4:
        return "Simple"
    elif vertices <= 7:
        return "Moderate"
    else:
        return "Complex"

# ================= MAIN =================
if uploaded:

    # ---------- LOAD IMAGE ----------
    image = np.array(Image.open(uploaded).convert("RGB"))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ---------- PREPROCESS ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if mode == "Shape Mode":
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(edges, kernel, iterations=1)
    else:
        edges = None
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    display = img.copy()
    results = []
    shape_crops = []

    # ---------- CONTOUR LOOP ----------
    for i, c in enumerate(contours, 1):
        area_px = cv2.contourArea(c)
        if area_px < 500:
            continue

        peri_px = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri_px, True)

        shape = classify_shape(c)
        complexity = shape_complexity(len(approx))

        area_cm = area_px * (scale ** 2)
        peri_cm = peri_px * scale

        M = cv2.moments(c)
        cx, cy = (0, 0)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        results.append([
            i, shape, complexity,
            round(area_cm, 2), round(peri_cm, 2)
        ])

        x, y, w, h = cv2.boundingRect(c)
        crop = img[y:y+h, x:x+w]
        shape_crops.append((i, crop))

        if show_contours:
            cv2.drawContours(display, [c], -1, (0, 255, 0), 2)

        if show_bbox:
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if show_centroid:
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

        label = f"ID {i}"
        cv2.rectangle(display, (cx+6, cy-18), (cx+60, cy-2), (0, 0, 0), -1)
        cv2.putText(display, label, (cx+8, cy-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ---------- IMAGE COMPARISON ----------
    st.subheader("Image Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("**Processed Image**")
        st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_column_width=True)

    # ---------- SHAPE COMPLEXITY EXPLANATION ----------
    st.markdown("---")
    st.subheader("Shape Complexity Interpretation")
    st.info(
        "**Simple:** ≤ 4 vertices  \n"
        "**Moderate:** 5 – 7 vertices  \n"
        "**Complex:** > 7 vertices"
    )

    # ---------- STEP-BY-STEP GRID ----------
    st.markdown("---")
    st.subheader("Step-by-Step Image Processing")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**Grayscale**")
        st.image(gray, use_column_width=True)

    with r1c2:
        st.markdown("**Blurred**")
        st.image(blur, use_column_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**Edge Detection**")
        st.image(edges if edges is not None else thresh, use_column_width=True)

    with r2c2:
        st.markdown("**Final Contours**")
        st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_column_width=True)

    # ---------- SHAPE GALLERY ----------
    st.markdown("---")
    st.subheader("Detected Objects Gallery")

    if shape_crops:
        cols = st.columns(min(4, len(shape_crops)))
        for idx, (obj_id, crop) in enumerate(shape_crops):
            with cols[idx % len(cols)]:
                st.image(crop, caption=f"ID {obj_id}", use_column_width=True)

    # ---------- TABLE ----------
    st.markdown("---")
    st.subheader("Geometric Feature Measurements")

    df = pd.DataFrame(
        results,
        columns=[
            "Object ID", "Shape Type", "Complexity",
            "Area (cm²)", "Perimeter (cm)"
        ]
    )

    st.dataframe(df.style.hide(axis="index"), use_container_width=True)

    # ---------- DISTRIBUTION GRAPHS ----------
    st.markdown("---")
    st.subheader("Shape Distribution Analysis")

    if not df.empty:
        counts = df["Shape Type"].value_counts()

        g1, g2 = st.columns(2)

        with g1:
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            ax1.bar(counts.index, counts.values)
            ax1.set_title("Shape Distribution")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        with g2:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90)
            ax2.set_title("Shape Proportion")
            ax2.axis("equal")
            st.pyplot(fig2)

else:
    st.info("Upload an image from the sidebar to begin analysis.")
