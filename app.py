import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;'>Shape & Contour Analyzer</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Geometric Shape Detection, Measurement & Feature Extraction Dashboard</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Detection Mode",
    ["Shape Mode", "Document Mode"],
    help=(
        "Shape Mode: Detects multiple independent geometric shapes.\n\n"
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
st.sidebar.markdown("🟢 Contours  |  🔵 Bounding Box  |  🔴 Centroid")

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

    image = np.array(Image.open(uploaded).convert("RGB"))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if mode == "Shape Mode":
        edges = cv2.Canny(blur, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(edges, kernel, iterations=1)
    else:
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display = img.copy()
    results = []

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

        results.append([i, shape, complexity, area_cm, peri_cm])

        if show_contours:
            cv2.drawContours(display, [c], -1, (0, 255, 0), 2)
        if show_bbox:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if show_centroid:
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

        # show only object ID to avoid clutter
        cv2.putText(display, f"ID {i}", (cx + 6, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Convert results to DataFrame
    df = pd.DataFrame(
        results,
        columns=["Object ID", "Shape Type", "Complexity", "Area (cm²)", "Perimeter (cm)"]
    )

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs(["🖼 Images", "📐 Measurements", "📊 Analytics"])

    # ---------- TAB 1: IMAGES ----------
    with tab1:
        st.subheader("Original vs Processed Image")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original Image**")
            st.image(image, use_column_width=True)
        with c2:
            st.markdown("**Processed Image**")
            st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_column_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Objects", len(df))
        m2.metric("Detection Mode", mode)
        m3.metric("Scale (cm/pixel)", scale)

    # ---------- TAB 2: MEASUREMENTS ----------
    with tab2:
        st.subheader("Geometric Feature Measurements")

        if not df.empty:
            st.dataframe(df.style.hide(axis="index"), use_container_width=True)

            st.download_button(
                "⬇ Download Measurements (CSV)",
                df.to_csv(index=False),
                "shape_measurements.csv",
                "text/csv"
            )
        else:
            st.warning("No shapes detected.")

    # ---------- TAB 3: ANALYTICS ----------
    with tab3:
        if not df.empty:
            st.subheader("Shape Summary")

            counts = df["Shape Type"].value_counts()

            k1, k2, k3 = st.columns(3)
            k1.metric("Most Common Shape", counts.idxmax())
            k2.metric("Total Shape Types", len(counts))
            k3.metric("Max Count", counts.max())

            st.markdown("---")

            # Bar chart
            fig1, ax1 = plt.subplots(figsize=(4.5, 3))
            ax1.bar(counts.index, counts.values)
            ax1.set_title("Shape Distribution")
            ax1.set_xlabel("Shape Type")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

            # Pie chart
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax2.set_title("Shape Proportion")
            ax2.axis("equal")
            st.pyplot(fig2)

            st.markdown("---")

            # Scatter plot: Area vs Perimeter
            st.subheader("Area vs Perimeter Analysis")

            fig3, ax3 = plt.subplots(figsize=(5, 4))
            for shape in df["Shape Type"].unique():
                subset = df[df["Shape Type"] == shape]
                ax3.scatter(subset["Area (cm²)"], subset["Perimeter (cm)"], label=shape)

            ax3.set_xlabel("Area (cm²)")
            ax3.set_ylabel("Perimeter (cm)")
            ax3.set_title("Area vs Perimeter Relationship")
            ax3.legend()
            st.pyplot(fig3)

        else:
            st.warning("No data available for analytics.")

else:
    st.info("Upload an image from the sidebar to begin analysis.")
