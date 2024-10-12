import os
import random
import json
from pycocotools.coco import COCO
import shutil

coco_annotations_path = './COCO/annotations/instances_train2017.json'
coco_captions_path = './COCO/annotations/captions_train2017.json'
coco_keypoints_path = './COCO/annotations/person_keypoints_train2017.json'
coco_image_path = './COCO/train2017/'

output_annotation_path = './COCO_10/mini_coco_annotations.json'
output_image_path = './COCO_10/train'

desired_num_image = 10

coco_instances = COCO(coco_annotations_path)
coco_captions = COCO(coco_captions_path)
coco_keypoints = COCO(coco_keypoints_path)

image_ids = coco_instances.getImgIds()

sampled_image_ids = random.sample(image_ids, desired_num_image)

categories = coco_instances.loadCats(coco_instances.getCatIds())
category_mapping = {cat['id']: cat['name'] for cat in categories} 

mini_coco_annotations = {
    "images": [],          
    "annotations": [],     
    "captions": [],        
    "keypoints": [],       
    "categories": categories  
}

os.makedirs(output_image_path, exist_ok=True)

for img_id in sampled_image_ids:
    img_info = coco_instances.loadImgs([img_id])[0]
    
    img_filename = img_info['file_name']
    src_img_path = os.path.join(coco_image_path, img_filename)
    dst_img_path = os.path.join(output_image_path, img_filename)
    
    shutil.copy(src_img_path, dst_img_path)
    mini_coco_annotations['images'].append(img_info)
    
    ann_ids = coco_instances.getAnnIds(imgIds=[img_id])
    annotations = coco_instances.loadAnns(ann_ids)
    
    for annotation in annotations:
        category_id = annotation['category_id']
        annotation['category_name'] = category_mapping[category_id]  # Add the category name to the annotation
    
    mini_coco_annotations["annotations"].extend(annotations)
    
    caption_ids = coco_captions.getAnnIds(imgIds=[img_id])
    captions = coco_captions.loadAnns(caption_ids)
    mini_coco_annotations['captions'].extend(captions)
    
    keypoint_ids = coco_keypoints.getAnnIds(imgIds=[img_id])
    keypoints = coco_keypoints.loadAnns(keypoint_ids)
    mini_coco_annotations['keypoints'].extend(keypoints)

# Save the mini COCO annotations to a JSON file
with open(output_annotation_path, 'w') as f:
    json.dump(mini_coco_annotations, f)

print(f"Smaller COCO dataset created with {desired_num_image} images.")
print(f"Annotations saved to {output_annotation_path}")
print(f"Images saved to {output_image_path}")