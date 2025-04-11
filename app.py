import streamlit as st
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Setup
st.title("🔍 Crime Scene Object Detector")
st.write("Upload an image and detect objects using your trained Faster R-CNN model.")

# Load model only once
@st.cache_resource
def load_model():
    num_classes = 14  # 13 objects + 1 background
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("fasterrcnn_crimescene.pth", map_location="cpu"))
    model.eval()
    return model

CLASS_LABELS = {
    1: 'blood',
    2: 'finger-print',
    3: 'glass',
    4: 'hammer',
    5: 'hand-gun',
    6: 'human-body',
    7: 'human-hair',
    8: 'human-hand',
    9: 'knife',
    10: 'null',
    11: 'rope',
    12: 'shoe-print',
    13: 'shotgun',
    14: 'victim'
}

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

    fig, ax = plt.subplots()
    ax.imshow(image)

    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        if score > threshold:
            x1, y1, x2, y2 = box
            class_name = CLASS_LABELS.get(label.item(), "Unknown")
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor='red', facecolor='none', linewidth=2))
            ax.text(x1, y1, f"{class_name} ({score:.2f})",
                    bbox=dict(facecolor='red', alpha=0.5), color='white')

    st.pyplot(fig)
