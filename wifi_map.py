import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from PIL import Image
import plotly.express as px

st.set_page_config(layout="wide", page_title="WiFi Signal Mapper")

st.title("📡 WiFi Signal Range Mapper")

def path_loss_model(d, K, gamma):
    return K - 10 * gamma * np.log10(d + 1e-9)

uploaded_file = st.sidebar.file_uploader("Upload Floorplan", type=["png", "jpg", "jpeg"])

if uploaded_file:
    bg_image = Image.open(uploaded_file)
    w, h = bg_image.size
    display_width = 800
    display_height = int(h * (display_width / w))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Map Surface")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=bg_image,
            height=display_height,
            width=display_width,
            drawing_mode="point",
            key="canvas",
        )

    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        if not objects.empty:
            with col2:
                st.subheader("Signal Data")

                df = objects[['left', 'top']].copy()
                df.columns = ['X', 'Y']

                dbm_inputs = []
                for i in range(len(df)):
                    val = st.number_input(f"Point {i+1} (dBm)", value=-50.0, key=f"p{i}")
                    dbm_inputs.append(val)

                df['dBm'] = dbm_inputs

                # NEW ✅ ignore invalid values (positive dBm)
                df['valid'] = df['dBm'] < 0

                ap_idx = st.selectbox("Which point is the Router?", range(1, len(df)+1)) - 1

                if st.button("🚀 Calculate & Show Range"):

                    # NEW ✅ filter valid points only
                    valid_df = df[df['valid']].copy()

                    if len(valid_df) < 2:
                        st.error("Need at least 2 valid (negative dBm) points.")
                    else:
                        ap_pos = np.array([df.iloc[ap_idx]['X'], df.iloc[ap_idx]['Y']])

                        meas_df = valid_df.drop(valid_df.index[valid_df.index == ap_idx], errors='ignore')

                        dists = np.sqrt(np.sum((meas_df[['X', 'Y']].values - ap_pos)**2, axis=1))
                        signals = meas_df['dBm'].values

                        popt, _ = curve_fit(path_loss_model, dists, signals, p0=[-30, 3.0])
                        K_fit, gamma_fit = popt

                        grid_x, grid_y = np.meshgrid(
                            np.linspace(0, display_width, 150),
                            np.linspace(0, display_height, 150)
                        )

                        grid_dists = np.sqrt((grid_x - ap_pos[0])**2 + (grid_y - ap_pos[1])**2)
                        grid_signals = path_loss_model(grid_dists, K_fit, gamma_fit)

                        st.success(f"Calculated Environment Factor (γ): {gamma_fit:.2f}")

                        # NEW 🎨 CUSTOM COLOR SCALE
                        custom_colors = [
                            [0.0, "grey"],      # -100
                            [0.25, "blue"],     # -80
                            [0.5, "yellow"],    # -70
                            [0.75, "green"],    # -60
                            [1.0, "red"]        # -30
                        ]

                        fig = px.imshow(
                            grid_signals,
                            color_continuous_scale=custom_colors,
                            range_color=[-90, -30],
                            labels={'color': 'dBm'}
                        )

                        # NEW ✅ add points + labels
                        fig.add_scatter(
                            x=df['X'],
                            y=df['Y'],
                            mode='markers+text',
                            marker=dict(color='black', size=8),
                            text=[str(i+1) for i in range(len(df))],
                            textposition="top center"
                        )

                        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("← Please upload a floorplan image in the sidebar.")
