from flask import Flask, send_from_directory, render_template, request
from flask_cors import CORS
import cv2
import numpy as np
import os
from random import randint

app = Flask(__name__)
CORS(app, support_credentials=True)

tasks = []

@app.route('/')
def hello_world():
    return render_template("image-detection.html")

@app.route('/predict', methods=["POST"])
def predict():
    random_id = randint(0,999)

    ori_file = request.files['ori_image']
    pure_name = ori_file.filename.split(".")[0]
    extension = ori_file.filename.split(".")[1]
    ori_img_new = f"{pure_name}_{random_id}.{extension}"
    ori_path = os.path.join("images", ori_img_new)
    ori_file.save(ori_path)

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    gpu = 0
    configPath = 'yolov4.cfg'
    weightsPath = 'yolov4.weights'
    image_path = "images/" + ori_img_new
    class_path = "coco.names"

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    if gpu == 1:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    image = cv2.imread(image_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weightsPath, configPath)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    pred_path = os.path.join("images", "predict_{}_{}.jpg".format(pure_name, random_id))
    cv2.imwrite(pred_path, image)

    if not tasks:
        id = 1
    else:
        id = tasks[-1]["id"] + 1

    data = {
        "id" : id,
        "original-image" : ori_img_new,
        "predicted-image" : "predict_{}_{}.jpg".format(pure_name, random_id)
    }
    tasks.append(data)

    return str(id)

@app.route('/predicted-image/<int:taskId>', methods=['GET'])
def pred_image(taskId):
    task = [task for task in tasks if task["id"] == taskId]
    return send_from_directory("images", task[0]["predicted-image"], cache_timeout=0)

@app.route('/ori-image/<int:taskId>', methods=['GET'])
def ori_image(taskId):
    task = [task for task in tasks if task["id"] == taskId]
    return send_from_directory("images", task[0]["original-image"], cache_timeout=0)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")