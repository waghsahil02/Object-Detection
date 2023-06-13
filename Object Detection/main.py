from pydoc import classname
import cv2
import pyttsx3

thres = 0.45
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pdtxt'
weightsPath = 'frozen_interface_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.InputScale(1.0 / 127.5)
net.InputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

engine = pyttsx3.init()

def getObjects(img, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.1)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                object_text = f"{className} detected"
                engine.say(object_text)
                engine.runAndWait()
    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, objects=['person', 'car', 'scissors', 'glove', 'knife', 'bottle', 'plate'])
        print(objectInfo)

        cv2.imshow("Object", img)
        cv2.waitKey(1)
