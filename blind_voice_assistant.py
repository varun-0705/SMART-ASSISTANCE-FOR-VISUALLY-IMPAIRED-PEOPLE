import cv2
import numpy as np
import pyttsx3
language = 'en'

cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = "coco.names"
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))

modelConfiguration = "yolov2.cfg"
modelWeights = "yolov2.weights"
engine = pyttsx3.init()

net = cv2.dnn.readNet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
oldvr=[]
while True:
    vr=[]
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT),[0,0,0],crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]* wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bbox, confs,confThreshold,nmsThreshold)

    for i in indices:
        i = i
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255),2)
        labeld=classNames[classIds[i]]
        if labeld in oldvr:
            for j in range(0,len(oldvr)):
                if oldvr[j] == labeld:
                    print("label availabe",oldvr[j],labeld)
                    print("X axis",oldvr[j+1],"Now xvalue",x)
                    d=oldvr[j+1]-x
                    print("d value",d)
            if d >0.0 and d >10.0:
                print(labeld,"Moving to right direction")
                msg=labeld+"Moving to right direction"
                engine.say(msg)
                engine.runAndWait()
                msg=""
            if d <0.0 and d<-10:
                print(labeld,"Moving to left direction")
                msg=labeld+"Moving to left direction"
                engine.say(msg)
                engine.runAndWait()
                msg=""
        vr.append(classNames[classIds[i]])
        vr.append(x)
        vr.append(y)
        print(vr)
    oldvr=vr
    cv2.imshow("Surveillance Camera", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# cap.release()
cv2.destroyAllWindows()

