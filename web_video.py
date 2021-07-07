import gradio as gr
import torch
import cv2

# Images
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg', 'zidane.jpg')
torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/raw/master/data/images/bus.jpg', 'bus.jpg')

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # force_reload=True to update


def yolo(fn):
    print('start')
    # 'C:\\Users\\karao\\AppData\\Local\\Temp\\ayvz4oru.mp4'
    capture = cv2.VideoCapture(fn)
    if not capture.isOpened():
        print('failed capture.isOpened()')
        exit(-1)

    while True:  # while true, read the camera
        ret, frame = capture.read()
        if not ret:
            break
        results = model(frame)  # inference
        results.render()  # updates results.imgs with boxes and labels


    return fn

inputs = gr.inputs.Video(type='mp4', label="Original Image")
outputs = gr.outputs.Video(label="Output Image")

title = "YOLOv5"
description = "YOLOv5 demo for object detection. Upload an image or click an example image to use."
article = "<p style='text-align: center'>YOLOv5 is a family of compound-scaled object detection models trained on the COCO dataset, and includes " \
          "simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, " \
          "and export to ONNX, CoreML and TFLite. <a href='https://github.com/ultralytics/yolov5'>Source code</a> |" \
          "<a href='https://apps.apple.com/app/id1452689527'>iOS App</a> | <a href='https://pytorch.org/hub/ultralytics_yolov5'>PyTorch Hub</a></p>"

examples = [['zidane.jpg'], ['bus.jpg']]
gr.Interface(yolo, inputs, outputs, title=title, description=description, article=article, examples=examples, analytics_enabled=False).launch(
    debug=True)