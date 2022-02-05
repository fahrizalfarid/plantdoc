# Inference

## opencv
```python
import numpy as np
import cv2
import os


labels = []


net = cv2.dnn.readNet('best.onnx')

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("class.names", "r") as f:
    labels = [cname.strip() for cname in f.readlines()]
f.close()

INPUT_WIDTH = 1280
INPUT_HEIGHT = 1280
SCORE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


def postprocess(image, predictions):
    confidences = []
    class_ids = []
    boxes = []

    image_width, image_height, _ = image.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    rows = predictions.shape[0]
    print(rows)

    for r in range(rows):
        row = predictions[r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD) 

    print(confidences,'conf')
    print(boxes,'boxes')

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]

        print(labels[class_id])

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, labels[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
    return image, result_class_ids



path = './results/'
imglist = os.listdir(path)

for img in imglist:
    image = cv2.imread(path+img)
    resized = cv2.resize(image, (INPUT_WIDTH,INPUT_HEIGHT), interpolation = cv2.INTER_AREA)  

    blob = cv2.dnn.blobFromImage(image , 1/255.0, 
                                (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()
    image_output, class_ids = postprocess(resized, predictions[0])
    if len(class_ids) > 0:
        cv2.imwrite("./results/%s_detection.png"%str(img), image_output)
        cv2.imshow("output", image_output)
        cv2.waitKey()
        cv2.destroyAllWindows()
```

## tflite (tensorflow==2.6.0)
```python
import tensorflow as tf
import numpy as np
import os
import cv2


imgsize = 1280
INPUT_WIDTH = 1280
INPUT_HEIGHT = 1280
SCORE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

labels = []


modelpath = './best-int8.tflite'
interpreter = tf.lite.Interpreter(model_path=modelpath, num_threads=12)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)


with open("class.names", "r") as f:
    labels = [cname.strip() for cname in f.readlines()]
f.close()



def preproces(imgfile):
    img = cv2.imread(imgfile)
    img = cv2.resize(img, (imgsize,imgsize), interpolation = cv2.INTER_AREA)
    x = np.expand_dims(img, axis=0)
    x = x.astype(np.uint8)
    return x,img


def postprocess(image, predictions):
    confidences = []
    class_ids = []
    boxes = []

    output_data = predictions[0]

    print(predictions.shape)

    image_width, image_height, _ = image.shape
    x_factor = image_width / imgsize
    y_factor =  image_height / imgsize

    rows = output_data.shape[0]

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .45):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, labels[class_id], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
    return image



path = './results/'
imglist = os.listdir(path)

for img in imglist:
    x, img = preproces(path+img)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    output_image = postprocess(img, predictions)
    cv2.imshow("output", output_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
```
