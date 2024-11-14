from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import torch

# Load the image from the local directory
image_path = 'Noised_Image/Original/adversarial_image_160009.png'
image = Image.open(image_path)

# Initialize the feature extractor and model
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-small')

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)

# Model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes

# Print logits and bounding boxes to verify output
print("Logits:", logits)
print("Bounding Boxes:", bboxes)