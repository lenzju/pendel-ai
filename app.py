import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt

st.title("KI-Pendel Analyse")

uploaded_file = st.file_uploader("Video hochladen", type=["mp4", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0,120,70])
        upper_red = np.array([10,255,255])

        mask = cv2.inRange(hsv, lower_red, upper_red)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            cx = x + w//2

            positions.append(cx)

    cap.release()

    st.success("Analyse abgeschlossen!")

    # Diagramm
    fig, ax = plt.subplots()
    ax.plot(positions)
    ax.set_title("Pendelbewegung")
    st.pyplot(fig)
