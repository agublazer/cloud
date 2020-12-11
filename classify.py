# pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
import boto3
import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
from collections import Counter
import requests
import json
from flask import Flask, render_template, request

app = Flask(__name__)
s3 = boto3.client('s3')


img_tranforms = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
	])


def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def classify_image(model, img):
	img = img_tranforms(img).unsqueeze(0)
	cl = model(img).argmax().item()
	return cl


def classify_video(model, video_path):
	results = []
	vidcap = cv2.VideoCapture(video_path)
	success, image = vidcap.read()
	count = 0
	while success:
		if count % 10 == 0:
			im_pil = Image.fromarray(image)
			results.append(classify_image(model, im_pil))
			success, image = vidcap.read()
		count += 1
	return most_frequent(results)

@app.route("/", methods=['GET'])
def hello():
    return "Hello from Python!"


@app.route("/", methods=['POST'])
def index():
	file_name = request.get_json(force=True)
	print('file name', file_name)
	s3.download_file('cloud-ucsp', file_name, file_name)
	# s3.download_file('cloud-ucsp', 'resnet34.pt', 'resnet34.pt')
	model = torch.jit.load('resnet34.pt')
	result = classify_video(model, file_name)
	s3.upload_file(file_name, 'cloud-ucsp', file_name[:-4] + '-' + str(result) + '.mp4')
	os.remove(file_name)
	# os.remove('resnet34.pt')
	s3.delete_object(Bucket='cloud-ucsp', Key=file_name)
	return {'statusCode': 200, 'body': result}

if __name__ == "__main__":
    app.run(host='0.0.0.0')
