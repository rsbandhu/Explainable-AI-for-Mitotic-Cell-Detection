import streamlit as st
import numpy as np
#from explainer.mitotic_classifier_explainer import get_explainers
#from explainer.explainer_plots import visualize_image_attr, plot_attributions

from explainer.mitotic_detections_summary import generate_mitotic_count_10hpf
from explainer.explainer_plots import visualize_image_attr, plot_attributions, plot_count_heatmap


all_mitotic_detections = st.session_state["all_mitotic_detections"]
wsi_state_dict = st.session_state["wsi_state_dict"]

patch_size = wsi_state_dict["patch_size"]
dims = wsi_state_dict["level_dims"] 
microns_per_pixel = wsi_state_dict["microns_per_pixel"]
img = st.session_state["img_lowres"]

#@st.cache(suppress_st_warning=True)
def plot_mitotic_in_10HPF(all_mitotic_detections, img, patch_size, dims, microns_per_pixel):
        
    detections_10hpf = generate_mitotic_count_10hpf(
        all_mitotic_detections, dims[0], dims[-1], patch_size=patch_size, microns_per_pixel=microns_per_pixel)
    #data_clsf_dir

    max_count = np.max(detections_10hpf)
    if (max_count > 7):
        target_text = f"High mitotic count in 10 consecutive HPF: {max_count}, greater than threshold of normal value 7. This is considered to be high grade tumor"

    else:
        target_text = f"Mitotic Count is normal. Max count in 10 consecutive HPF is {max_count}"
    
    fig = plot_count_heatmap(img, detections_10hpf)

    st.session_state["fig_10hpf"] = fig
    st.session_state["10hpf_text"] = target_text

    print("done creating heatmaps", detections_10hpf.shape)

st.header("Mitotic Count map in 10 consecutive High Power Field (Area ~ 2.37 sq mm)")
if "fig_10hpf" not in st.session_state or "10hpf_text" not in st.session_state:
    plot_mitotic_in_10HPF(all_mitotic_detections, img, patch_size,
                      dims, microns_per_pixel)

st.write(st.session_state["fig_10hpf"])
st.markdown(
    f'<h3 style="color:#0000FF;font-size:20px;">{st.session_state["10hpf_text"]}</h3>', unsafe_allow_html=True)
