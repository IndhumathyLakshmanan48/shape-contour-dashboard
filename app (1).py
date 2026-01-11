import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Shape & Contour Analyzer", layout="wide")

st.markdown("<h1 style='text-align:center;'>🧠 Smart Shape & Contour Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Interactive Computer Vision Dashboard</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙ Controls")
uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])
show_contours = st.sidebar.checkbox("Show Contours", True)
show_bbox = st.sidebar.checkbox("Show Bounding Box", True)
show_centroid = st.sidebar.checkbox("Show Centroid", True)
scale = st.sidebar.slider("Pixel → cm Scale", 0.01, 1.0, 0.1)

def classify_shape(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    v = len(approx)
    if v == 3: return "Triangle"
    if v == 4:
        x,y,w,h = cv2.boundingRect(approx)
        return "Square" if 0.95 < w/h < 1.05 else "Rectangle"
    if v == 5: return "Pentagon"
    if v > 6: return "Circle"
    return "Irregular"

if uploaded:
    image = np.array(Image.open(uploaded).convert("RGB"))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display = img.copy()
    shapes = []

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        shape = classify_shape(c)
        shapes.append(shape)

        if show_contours:
            cv2.drawContours(display, [c], -1, (0,255,0), 2)

        if show_bbox:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(display, (x,y), (x+w,y+h), (255,0,0), 2)

        if show_centroid:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv2.circle(display, (cx,cy), 5, (0,0,255), -1)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Processed Image")
        st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.subheader("Metrics")
        st.metric("Total Objects", len(shapes))
        st.write("Detected Shapes:", shapes)

    st.subheader("Shape Analytics")

if len(shapes) > 0:
    unique_shapes = list(dict.fromkeys(shapes))  # preserves order
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
    st.info("Upload an image from the sidebar")
