import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_coco_data(coco_annotations_path):
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def prepare_inputs(image_path, labels, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=labels, return_tensors="pt", padding=True, truncation=True).to(device)
    return inputs, image

def get_label_weights(logits, labels):
    # Get softmax probabilities (weights) for each label
    weights = torch.softmax(logits, dim=1).squeeze().cpu().detach().numpy()
    return weights

def plot_histogram(labels, original_weights, adversarial_weights, image_id, output_dir):
    indices = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(indices - 0.2, original_weights, 0.4, label='Original Image', color='red')
    plt.bar(indices + 0.2, adversarial_weights, 0.4, label='Adversarial Image', color='blue')

    plt.xticks(indices, labels, rotation=90)
    plt.xlabel('Labels')
    plt.ylabel('Weights')
    plt.legend()
    plt.title('Weights Comparison Between Original and Adversarial Images')

    # Save the histogram as a PNG file
    histogram_path = os.path.join(output_dir, f"histogram_{image_id}.png")
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()  # Close the plot to free up memory

def generate_adversarial_image(model, inputs, true_label_idx, epsilon):
    criterion = nn.CrossEntropyLoss()

    inputs['pixel_values'].requires_grad = True

    # Forward pass
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image

    # True label selected by argmax
    true_label = torch.tensor([true_label_idx]).to(device)

    loss = criterion(logits_per_image, true_label)

    model.zero_grad()
    loss.backward()

    data_grad = inputs['pixel_values'].grad.data

    sign_data_grad = data_grad.sign()
    perturbed_image = inputs['pixel_values'] + epsilon * sign_data_grad

    # Ensure pixel values are within [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

def display_and_save_images(image, perturbed_image, image_id, output_dir):
    perturbed_image_np = perturbed_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title("Adversarial Image")
    plt.imshow(perturbed_image_np)
    plt.show()

    adversarial_image_pil = Image.fromarray((perturbed_image_np * 255).astype('uint8'))
    adversarial_image_path = os.path.join(output_dir, f"adversarial_image_{image_id}.png")
    adversarial_image_pil.save(adversarial_image_path)

def log_labels(image_id, original_label, adversarial_label, output_dir):
    log_file = os.path.join(output_dir, "labels_log.txt")
    
    with open(log_file, "a") as f:
        f.write(f"Image ID: {image_id}\n")
        f.write(f"Original Label: {original_label}\n")
        f.write(f"Adversarial Label: {adversarial_label}\n")
        f.write("------------------------------------------------\n")

def generate_pie_chart(success_count, fail_count, output_dir):
    labels = 'Success', 'Failure'
    sizes = [success_count, fail_count]
    colors = ['lightgreen', 'lightcoral']
    explode = (0.1, 0)  # explode the 1st slice (Success)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title('Attack Success vs Failure')
    plt.savefig(os.path.join(output_dir, 'attack_pie_chart.png'))
    plt.close()
    

def generate_adversarial_images(coco_data, model, processor, coco_image_path, epsilon, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    success_count=0
    fail_count=0
    result_data={"images":[]}

    for image_info in coco_data['images']:
        image_path = os.path.join(coco_image_path, image_info['file_name'])
        image_id = image_info['id']

        labels = [f"a {annotation['category_name']}" for annotation in coco_data['annotations'] if annotation['image_id'] == image_id]

        if not labels:
            continue  # Skip images with no labels

        inputs, image = prepare_inputs(image_path, labels, processor)
        
        with torch.no_grad():
            original_outputs = model(**inputs)
        logits_per_image = original_outputs.logits_per_image

        original_weights = get_label_weights(logits_per_image, labels)
        true_label_idx = logits_per_image.argmax().item()
        original_label = labels[true_label_idx]

        # Generate adversarial image
        perturbed_image = generate_adversarial_image(model, inputs, true_label_idx, epsilon)

        # Classify the adversarial image
        with torch.no_grad():
            adversarial_inputs = {'pixel_values': perturbed_image, 'input_ids': inputs['input_ids']}
            adversarial_outputs = model(**adversarial_inputs)


        adversarial_logits = adversarial_outputs.logits_per_image
        adversarial_weights = get_label_weights(adversarial_logits, labels)

        adversarial_label_idx = adversarial_logits.argmax().item()
        adversarial_label = labels[adversarial_label_idx]

        if original_label != adversarial_label:
            success_count +=1
        else:
            fail_count+=1
        
        result_data["images"].append({
            "id": image_id,
            "file_name": image_info['file_name'],
            "original_label": original_label,
            "adversarial_label":adversarial_label
        })

        log_labels(image_id, original_label, adversarial_label, output_dir)

        display_and_save_images(image, perturbed_image, image_id, output_dir)
        # Plot and save the histogram of weights
        plot_histogram(labels, original_weights, adversarial_weights, image_id, output_dir)
    
    generate_pie_chart(success_count, fail_count,output_dir)
    
    with open(os.path.join(output_dir,"adversarial_result.json"),'w') as f:
        json.dump(result_data,f,indent=4)


def main():
    parser = argparse.ArgumentParser(description="Adversarial attack generation using CLIP model.")
    parser.add_argument('--e', type=float, required=True, help="Perturbation size for adversarial attack")
    parser.add_argument('--annPath', type=str, required=True, help="Path to COCO annotations file")
    parser.add_argument('--imgPath', type=str, required=True, help="Path to COCO images directory")
    parser.add_argument('--outDir', type=str, required=True, help="Directory to save adversarial images")
    
    args = parser.parse_args()

    model, processor = load_clip_model()
    coco_data = load_coco_data(args.annPath)
    
    generate_adversarial_images(coco_data, model, processor, args.imgPath, args.e, args.outDir)

if __name__ == "__main__":
    main()