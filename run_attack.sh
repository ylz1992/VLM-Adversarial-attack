#!/bin/bash

# Configuration parameters
EPSILON=0.05
COCO_ANNOTATIONS_PATH='./COCO_10/mini_coco_annotations.json'
COCO_IMAGE_PATH='./COCO_10/train'
OUTPUT_DIR='./COCO_10/adversarial_images'

# Run the adversarial attack Python script
python3 adversarial_attack.py --e $EPSILON --annPath $COCO_ANNOTATIONS_PATH --imgPath $COCO_IMAGE_PATH --outDir $OUTPUT_DIR