import torch
import os
import json
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Initialize the model and processor
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r101vd_coco_o365")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r101vd_coco_o365")

# Define the directory containing images and the output JSON file
image_directory = "Noised_Image/Original/"
output_file = os.path.join(image_directory, "rtdetr_output.json")

# List to store detection results for all images
all_detections = []

# Process each image in the directory
for filename in os.listdir(image_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
        image_path = os.path.join(image_directory, filename)
        
        # Open and process the image
        image = Image.open(image_path)
        inputs = image_processor(images=image, return_tensors="pt")
        
        # Run the model and get the output
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process to get detected objects with a threshold
        results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

        # Collect detection data for this image
        image_detections = {"filename": filename, "detections": []}
        for result in results:
            for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                score, label = score.item(), label_id.item()
                box = [round(i, 2) for i in box.tolist()]  # Convert to list of rounded values
                label_name = model.config.id2label[label]
                detection = {
                    "label": label_name,
                    "score": round(score, 2),
                    "box": {"xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3]}
                }
                image_detections["detections"].append(detection)
        
        # Append the detections for this image to the main list
        all_detections.append(image_detections)
        print(f"Processed {filename}")

# Save all detections to a single JSON file
with open(output_file, "w") as f:
    json.dump(all_detections, f, indent=4)

print(f"All detections saved to {output_file}")