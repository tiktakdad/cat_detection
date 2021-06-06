import gradio as gr
import numpy as np
import torch
import cv2
from PIL import Image


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # force_reload=True to update

def snap(image):
    return np.flipud(image)



title = "YOLOv5"
description = "YOLOv5 demo for object detection. Upload an image or click an example image to use."
article = "<p style='text-align: center'>YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes " \
          "simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, " \
          "and export to ONNX, CoreML and TFLite. <a href='https://github.com/ultralytics/yolov5'>Source code</a> |" \
          "<a href='https://apps.apple.com/app/id1452689527'>iOS App</a> | <a href='https://pytorch.org/hub/ultralytics_yolov5'>PyTorch Hub</a></p>"
examples = [['zidane.jpg'], ['bus.jpg']]

catday = gr.Interface(
    snap,
    gr.inputs.Image(source="webcam", tool=None),
    gr.outputs.Image(type="pil", label="Output Image"),
    title=title, description=description, article=article, examples=examples, analytics_enabled=False
)

if __name__ == "__main__":
    catday.launch()