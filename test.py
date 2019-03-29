import os
import cv2
import time
import numpy as np

memory = {}
line = [(43, 543), (550, 655)]
counter = 0
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def ccw(A,B,C):
    print((C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0]))

    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# yolo label 가져오기
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#print("[INFO] loading YOLO from disk...")
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
print(ln)
print(net.getUnconnectedOutLayers())
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(ln)

vs = cv2.VideoCapture("input/highway.mp4")
writer = None
(W, H) = (None, None)

frameIndex = 0

(grabbed, frame) = vs.read()

blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)

net.setInput(blob)

start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# 검출된 bounding boxes, confidences, class IDs 각각 초기화
boxes = []
confidences = []
classIDs = []

while True:
    # frame을 계속 읽어나간다.
    (grabbed, frame) = vs.read()

    # 만약 frame을 잡지 못하면 종료
    if not grabbed:
        break

    # frame의 공간 크기가 존재하지않으면 만들어준다
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # 입력 frame에서 blob(대용량 바이너리 객체)을 구하고 forward를 수행한다.
    # YOLO 객체 검출을 통과하면 bounding box와 관련 확률을 제공한다.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # 검출된 bounding boxes, confidences, class IDs 각각 초기화
    boxes = []
    confidences = []
    classIDs = []

    # 각 output에 대해 반복
    for output in layerOutputs:
        # 각 output에서 각각 검출된 것에 대해 반복
        for detection in output:
            # 검출된 것에 대한 class IDs, confidence(확률) 추출
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # 최소 확률보다 큰 확률을 가지는 취약한 예측을 걸러낸다
            if confidence > 0.5:
                # YOLO가 실제로 경계 상자의 중심(x, y) 좌표에 이어
                # 상자의 폭과 높이를 반환한다는 점을 염두에 두고,
                # 이미지 크기에 비례하여 경계 상자 좌표를 다시 조정한다.
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # 중앙 (x,y)를 사용하여 bounding box의 왼쪽 위 모서리 구함
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # bounding box coordinates,confidences, class IDs 추가
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    print("boxs: ",boxes)
    print("confidences: ",confidences)

    # 겹치는 bounding boxe를 위한 non-maxima suppression 사용
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    print(idxs.flatten())

    dets = []
    if len(idxs) > 0:
        # 우리가 가지는 지표 만큼 반복
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)

    from sort import *

    tracker = Sort()

    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    print(dets)
    print(tracks)
    print("box : ",boxes)
    print("ID : ",indexIDs)
    print("memory : ",memory)

    break