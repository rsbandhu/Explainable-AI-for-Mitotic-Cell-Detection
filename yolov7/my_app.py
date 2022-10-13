"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np

import time
import os

df = pd.DataFrame({
  'Parameter': ['Total Mitotic Detections', 'Total Mitotic Look Alike Detections', 
  'ProPhase Count', 'Metaphase count', 'Anaphase count'],
  'Value': [10, 20, 30, 40, 50]
})

st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")

st.write("Here's our first attempt at using data to create a table:")
st.write(df)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Upload your image ❄️")

'Starting a long computation...'

# Add a placeholder
#latest_iteration = st.empty()
#bar = st.progress(0)

""" @st.cache
def show_progress(n):
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(n):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    '...and now we\'re done!'

show_progress(40)
 """


st.title("Upload your Images")

menu = ["Image"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Image":
    st.subheader("Upload Images")
    image_file = st.file_uploader("Upload an Image", type=["svs", "jpeg"])
    if image_file is not None:
        # see image details
        file_details = {"FileName":image_file.name, "FileType":image_file.type}

        #Saving image file in a WSI folder
        with open(os.path.join("WSI",image_file.name),"wb") as f:
            f.write(image_file.getbuffer())
        
        st.success("WSI image has been uploaded")

slide_path = str(os.path.join('WSI',image_file.name))
slide = openslide.open_slide(str(slide_path))
print(f"size of slide: {slide.dimensions}")
disp_size = (44999, 47721)
img = slide.read_region((0, 0), level=0, size=disp_size)
print(type(img))
st.image(img)

