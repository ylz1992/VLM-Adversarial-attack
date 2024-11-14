import requests
import os
import json

API_URL = "https://api-inference.huggingface.co/models/hustvl/yolos-small"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def process(path,output_file):
    all_outputs = []
    for filename in os.listdir(path):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(path,filename)
            output = query(file_path)
            
            all_outputs.append({"filename":filename, "output":output})
            
    with open(output_file,'w') as f:
        json.dump(all_outputs,f,indent=4)
            
            
            
path = "Noised_Image/Original/"
output_file = os.path.join(path,"yolo_ouput.json")
process(path,output_file)