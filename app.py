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
    "<h2 style='text-align:center;'>Shape & Contour Analyzer</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Geometric Shape Detection, Measurement & Feature Extraction</p>",
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

uploaded = st.sidebar.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

show_contours = st.sidebar.checkbox("Show Contours", True)
show_bbox = st.sidebar.checkbox("Show Bounding Box", True)
show_centroid = st.sidebar.checkbox("Show Centroid", True)

scale = st.sidebar.slider("Pixel → cm Scale", 0.01, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.caption("Green: Contours | Blue: Bounding Box | Red: Centroid")

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
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    display = img.copy()
    results = []

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
        cx, cy = 0, 0
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        results.append([
            i, shape, complexity,
            round(area_cm, 2), round(peri_cm, 2)
        ])

        if show_contours:
            cv2.drawContours(display, [c], -1, (0, 255, 0), 2)

        if show_bbox:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if show_centroid:
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

        # Minimal label to avoid clutter
        cv2.putText(
            display, f"ID {i}", (cx + 6, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )

    # ---------- IMAGE SECTION ----------
    st.subheader("Image Comparison")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.markdown("**Original Image**")
        st.image(image, use_column_width=True)

    with img_col2:
        st.markdown("**Processed Image**")
        st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_column_width=True)

    # ---------- METRICS ----------
    st.markdown("### Summary Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Objects", len(results))
    m2.metric("Mode", mode)
    m3.metric("Scale (cm/pixel)", scale)

    # ---------- TABLE ----------
    st.markdown("---")
    st.subheader("Geometric Feature Measurements")

    if results:
        df = pd.DataFrame(
            results,
            columns=[
                "Object ID", "Shape Type", "Complexity",
                "Area (cm²)", "Perimeter (cm)"
            ]
        )
        st.dataframe(df.style.hide(axis="index"), use_container_width=True)

        st.download_button(
            "Download Measurements (CSV)",
            df.to_csv(index=False),
            "shape_measurements.csv",
            "text/csv"
        )
    else:
        st.warning("No valid shapes detected.")

    # ---------- ANALYTICS ----------
    if results:
        st.markdown("---")
        st.subheader("Shape Analytics")

        counts = df["Shape Type"].value_counts()

        # Small KPI row
        k1, k2, k3 = st.columns(3)
        k1.metric("Most Common Shape", counts.idxmax())
        k2.metric("Total Shape Types", len(counts))
        k3.metric("Max Count", counts.max())

        # Small charts row
        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            ax1.bar(counts.index, counts.values)
            ax1.set_title("Shape Distribution")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.0f%%",
                startangle=90
            )
            ax2.set_title("Shape Proportion")
            ax2.axis("equal")
            st.pyplot(fig2)

        # Scatter plot (compact)
        st.markdown("### Area vs Perimeter Relationship")

        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        for shape in df["Shape Type"].unique():
            subset = df[df["Shape Type"] == shape]
            ax3.scatter(
                subset["Area (cm²)"],
                subset["Perimeter (cm)"],
                label=shape
            )

        ax3.set_xlabel("Area (cm²)")
        ax3.set_ylabel("Perimeter (cm)")
        ax3.legend(fontsize=8)
        st.pyplot(fig3)

else:
    st.info("Upload an image from the sidebar to begin analysis.")
