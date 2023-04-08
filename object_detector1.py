import cv2

# Open the webca
cap = cv2.VideoCapture(0)


classLabels = [ "person", "bicycle", "car", "motorcycle",
  "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
  "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
  "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
  "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
  "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
  "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN


while True:
    success, frame = cap.read()

    ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.5)

    if len(ClassIndex) != 0:
        for classInd, confidence, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if (classInd<=80):
                cv2.rectangle(frame, boxes, color= (0, 255, 0), thickness=2)
                cv2.putText(frame, classLabels[classInd - 1].upper(), (boxes[0] + 10, boxes[1] + 30), font, fontScale=font_scale,
                            color=(0, 255, 0), thickness=3)
                cv2.putText(frame, str(round(confidence*100,2)), (boxes[0]+200,boxes[1]+30), font, fontScale=font_scale,
                            color=(0, 255, 0), thickness=3)

    cv2.imshow("Output", frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





