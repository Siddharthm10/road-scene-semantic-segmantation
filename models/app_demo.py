import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import numpy as np
import time
import cv2

from deeplabv3_attention import DeepLabV3PlusWithAttention

# Model config
NUM_CLASSES = 19
MODEL_PATH = "runs/deeplabv3plus_with_attention.pth" 

# Load model
@st.cache_resource
def load_model():
    model = DeepLabV3PlusWithAttention(num_classes=NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Cityscapes 19-class color palette
def create_color_map():
    return np.array([
        [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
        [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
        [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
        [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
    ])

# Inference function
def segment_image(image, model):
    transform = T.Compose([
        T.Resize((512, 1024)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        upsampled = F.interpolate(output, size=(image.height, image.width), mode='bilinear', align_corners=False)
        seg = upsampled.argmax(dim=1)[0].cpu().numpy()
    return seg

# Apply colormap
def apply_color_map(segmentation, color_map):
    return color_map[segmentation]

# UI
st.title("Semantic Segmentation with Custom DeepLabV3 (Cityscapes - 19 Classes)")

model = load_model()
color_map = create_color_map()

uploaded_file = st.file_uploader("Upload an image (preferably ≤1024x2048)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("Running segmentation..."):
        start_time = time.time()
        segmentation = segment_image(image, model)
        elapsed_time = time.time() - start_time

    # Apply color map
    color_mask = apply_color_map(segmentation, color_map)
    mask_image = Image.fromarray(color_mask.astype(np.uint8))

    st.image(mask_image, caption="Segmented Output", use_container_width=True)
    st.markdown(f"**Processing Time:** {elapsed_time:.2f} seconds")
