import cv2
import numpy as np    # Numerical python to create an array

modelConfiguration="cfg/yolov3.cfg"
modelWeights="yolov3.weights"

yoloNetwork=cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
print(yoloNetwork)

labels= open("coco.names").read().strip().split('\n')
print(labels)

image=cv2.imread('static/img1.jpg')
image=cv2.resize(image, (700,500))

dimensions=image.shape[:2]
print(dimensions)
H=dimensions[0]
W=dimensions[1]

confidenceThreshold=0.5
NMSThreshold=0.3



# Blob
blob = cv2.dnn.blobFromImage(image, 1/255, (416,416))

# Input the image blob to the model
yoloNetwork.setInput(blob)

# Get names of unconnected output layers
layerName=yoloNetwork.getUnconnectedOutLayersNames()
# print("layername: ", layerName)

# Forward the input data through network(neural network)
layerOutputs=yoloNetwork.forward(layerName)

#initiallize lists to store bounding, confidences and classId's
boxes=[]
confidences=[]
classIds=[]

for output in layerOutputs:
    #get class score and id of class with the highest score
    for detection in output:
        score=detection[5:]
        classId=np.argmax(score)
        confidence=score[classId]
        if confidence>confidenceThreshold:
            box=detection[0:4]*np.array([W,H,W,H])
            (centerX,centerY,width,height)=box.astype('int')
            x=int(centerX-(width)/2)
            y=int(centerY-(height)/2)

            boxes.append([x,y,int(width),int(height)])
            confidences.append(float(confidence))
            classIds.append(classId)

print(len(boxes))
indexes=cv2.dnn.NMSBoxes(boxes,confidences,confidenceThreshold,NMSThreshold)
font=cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(boxes)):
    if i in indexes:
        x=boxes[i][0]
        y=boxes[i][1]
        w=boxes[i][2]
        h=boxes[i][3]

        color=(0,0,255)
        label=labels[classIds[i]]
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        text= '{}:{:.2f}'.format(label,confidences[i]*100)
        cv2.putText(image,text,(x,y-5), font,0.5,color,2)


# Renders an image from an array
cv2.imshow("object", image)

# Display the window infinitely until any key press
cv2.waitKey(0)