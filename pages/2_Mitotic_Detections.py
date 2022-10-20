import time
import streamlit as st
import plotly.express as px

import openslide

from explainer.mitotic_classifier_explainer import get_explainers
from explainer.mitotic_detections_summary import detection_bbox_width_height, summarize_detections
from explainer.explainer_plots import plot_width_height, visualize_image_attr, plot_attributions
from image_utils.plot_detections import plot_all_detections_in_region, get_detection_locations, plot_detections_in_zoomed_region


data_clsf_dir = st.session_state["data_clsf_dir"]
class_names = st.session_state["class_names"]
all_mitotic_detections = st.session_state["all_mitotic_detections"]

wsi_state_dict = st.session_state["wsi_state_dict"]
patch_size = wsi_state_dict["patch_size"]
dims = wsi_state_dict["level_dims"] 
mag_ratio = wsi_state_dict["mag_ratio"]
save_img_path = wsi_state_dict["save_img_path"]

# Display all detections on top of WSI at low mag

st.header("Mitotic Detections on WSI (displayed at lowest magnification)")
st.write(f"**Yellow colors:** Detections that are classified in one of the 3 phases: ProPhase, Metaphase and Anaphase / Telophase")
st.write(f"**Blue colors:** Detections that can't be classified in one of the above 3 phases")

img = st.session_state["img_lowres"]
fig = px.imshow(img, width=1000, height=1000)
shapes = get_detection_locations(
    all_mitotic_detections, patch_size, mag_ratio=mag_ratio)

fig.update_layout(shapes=shapes)
st.plotly_chart(fig)

st.write("---")
st.header("Summary of mitotic Detections")

det_summary = summarize_detections(all_mitotic_detections, class_names)

st.write(det_summary)

# Plots distribution on bounding box widths and heights
det_widths, det_heights = detection_bbox_width_height(all_mitotic_detections)
fig = plot_width_height(det_widths, det_heights)
st.write(fig)

st.markdown(f'<h3 style="color:#0000FF;font-size:18px;" > 50 pixels correspond to 12.5 microns </h3 >',
            unsafe_allow_html=True)
#st.markdown("**Integrated Gradients:** ", unsafe_allow_html=False)

st.write("---")
st.subheader(f"Mitosis Phases of detections in the zoomed region")

st.markdown(f"**Number denotes the probability of being mitotic**")

# Zoom in some region to visualize high resolution image of detection region
# Add bounding box and labels
with st.sidebar:
    #if st.button('Choose the region '):
    w = 40
    h = 40
    x = st.number_input(
        f"Insert the x-coordinate of the top left corner. \n Must be between 0 and {dims[-1][0]}", min_value=0, max_value=dims[-1][0], value=1408)
    y = st.number_input(f"Insert the y-coordinate of the top left corner. \n Must be between 0 and {dims[-1][1]}",
                        min_value=0, max_value=dims[-1][1], value=2270)
    w = st.number_input("Specify the width of the region . Must be between 20 and 60",
                        min_value=20, max_value=60, value=40)

    h = w
    st.write("Zooming in the following region: ")
    st.write("Top left = ", x, y, "Bottom right : ", x+w, y+h)

w = int(w*mag_ratio)
h = w
print(f'w = {w}')
img_label, detections_in_box, imgs = plot_all_detections_in_region(
    x, y, w, h, all_mitotic_detections, save_img_path, patch_size, class_names)


### *****  This part is for fixing the text inside zoomed region using plotly chart
#img1, shapes, locs_labels = plot_detections_in_zoomed_region(
#    x, y, w, h, all_mitotic_detections, save_img_path, patch_size, class_names)

#fig = px.imshow(img1, width=640, height=640)
#fig.update_layout(shapes=shapes)
#st.plotly_chart(fig)

#for item in locs_labels:
#    top_left = item[0]
#    bot_right = item[1]
#   label = item[2]
#    detect_prob = str(item[3])

#    fig.add_annotation(x=top_left[0]-5, y=top_left[1]-5,
#                       text= label, height=20, showarrow=False)

fig = px.imshow(img_label, width=1000, height=1000)
st.plotly_chart(fig)

st.session_state["detections_in_box"] = detections_in_box
detections_in_box = st.session_state["detections_in_box"]

#@st.cache(suppress_st_warning=True)
def plot_attributions_in_zoomed_region(detections_in_box, imgs, data_clsf_dir, class_names):

    t0 = time.time()
    figs = []

    for img_file in detections_in_box:
        
        img = imgs[img_file]

        img_file = img_file + '.jpeg'
        print(f"processing file : {img_file}")

        img_original, saliency, attr_ig, attr_gradshap, attributions_occ = get_explainers(
            img_file, data_clsf_dir, class_names)

        original_image, norm_attr, cmap, vmin, vmax = visualize_image_attr(
            saliency, img_original,  sign="absolute_value",
            show_colorbar=True, title="Overlayed Gradient Magnitudes",
            use_pyplot=False)
        #fig, h = plot_image_attribution_heatmap(original_image, norm_attr)
        #st.write(fig)

        original_image, attr_ig, cmap, vmin, vmax = visualize_image_attr(
            attr_ig, img_original,  sign="all",
            show_colorbar=True, title="Overlayed Gradient Magnitudes",
            use_pyplot=False)

        original_image, attr_gradshap, cmap, vmin, vmax = visualize_image_attr(
            attr_gradshap, img_original,  sign="all",
            show_colorbar=True, title="Overlayed Gradient Shap Values",
            use_pyplot=False)
        #fig, h = plot_image_attribution_heatmap(original_image, norm_attr, cmap='PiYG')
        #st.write(fig)

        original_image, attr_occ, cmap, vmin, vmax = visualize_image_attr(
            attributions_occ, img_original,  sign="all",
            show_colorbar=True, title="Overlayed Gradient Magnitudes",
            use_pyplot=False)
        #fig, h = plot_image_attribution_heatmap(original_image, norm_attr, cmap='PiYG')

        attributions = [norm_attr, attr_ig, attr_gradshap, attr_occ]
        cmaps = ["Blues", "PiYG", "PiYG", "PiYG"]
        titles = ["Saliency Map", 'Integrated gradients',
                'Gradient Shap', "Image Occlusion"]
        
        fig = plot_attributions(original_image, attributions, cmaps, titles)
        #st.write(fig)

        #st.write(img_file, img.shape, saliency.shape, attr_ig.shape, attributions_occ.shape)

        figs.append([img_file, fig])
    print(f"Done processing all files")
    return figs

    print(f"Total time display heatmaps: {time.time() - t0}")


st.write("---")
st.header("Visual attributions of Mitotic Phase Classification")

st.markdown(
    f'<h1 style="color:#0000FF;font-size:24px;">Gradient based Methods:</h1>', unsafe_allow_html=True)
st.markdown("**Saliency Map** : Computes the gradient of the output with respect to input image pixel values.  ", unsafe_allow_html=False)
st.markdown("**Integrated Gradients** : Integral of the gradients when the input is varied along a line from baseline to target input image ", unsafe_allow_html=False)
st.markdown("**Gradient SHAP** : Integral of the gradients when the input is varied along a line from baseline to target input image ", unsafe_allow_html=False)


st.markdown(
    f'<h1 style="color:#0000FF;font-size:24px;">Occlusion based Methods: </h1>', unsafe_allow_html=True)

st.markdown("**Occlusion** : Replaces a rectangular region with a baseline and then computes the difference in output from original", unsafe_allow_html=False)

#if "fig_attributions" not in st.session_state:
fig_attributions = plot_attributions_in_zoomed_region(
    detections_in_box, imgs, data_clsf_dir, class_names)

st.session_state["fig_attributions"] = fig_attributions

#else:
if st.session_state["fig_attributions"]:
    for i, item in enumerate(st.session_state["fig_attributions"]):
        filename = item[0]
        fig = item[1]
        
        st.subheader(filename)
        st.write(fig)

