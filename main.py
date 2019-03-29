# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

# 기존에 있는 이미지 제거
files = glob.glob('output/*.png')
for f in files:
   os.remove(f)

# 실시간 추적 모듈 sort 사용
from sort import *
tracker = Sort()
memory = {}
# 카운팅하는 line
line = [(43, 543), (550, 655)]
counter = 0

# input,output,yolo weight argments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# AB와 CD가 교차하는 경우 True
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# yolo label 가져오기
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# 각 class 레이블을 나타내도록 색상목록 초기화
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# YOLO weights 과 model configuration의 경로
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
# YOLO 모델을 불러온다. 
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# YOLO 모델 layer 이름
ln = net.getLayerNames()
# OUTPUT layer 가져오기
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# 비디오파일 읽기
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

frameIndex = 0

# 비디오파일의 총 frame 수
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# 비디오파일의 총 frame 수를 찾는데 에러가 발생할 경우
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# 총 frame 만큼 반복문
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
			if confidence > args["confidence"]:
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

	# 겹치는 bounding boxe를 위한 non-maxima suppression 사용
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
	
	dets = []
	if len(idxs) > 0:
		# 우리가 가지는 지표 만큼 반복 (왼아래,오른아래,왼위,오른위,신뢰도)
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			dets.append([x, y, x+w, y+h, confidences[i]])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	dets = np.asarray(dets)
	# object tracking 시작
	# 정확도가 좋은 순서대로 정렬해준다.
	tracks = tracker.update(dets)

	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = {}

	# bounding box 좌표, ID
	for track in tracks:
		boxes.append([track[0], track[1], track[2], track[3]])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]

	if len(boxes) > 0:
		i = int(0)
		for box in boxes:
			# bounding box 좌표 추출
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))

			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

			# 검출 물체에 박스를 그리고 각 ID 마다 색을 다르게 한다.
			color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			cv2.rectangle(frame, (x, y), (w, h), color, 2)

			# 메모리에 저장된 이전 frame 줌심좌표와 현재 frame 중심좌표로
			# 선을 만들고 그 선과 count line이 교차할때 counting 한다.
			if indexIDs[i] in previous:
				previous_box = previous[indexIDs[i]]
				(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
				(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
				p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
				p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
				cv2.line(frame, p0, p1, color, 3)

				if intersect(p0, p1, line[0], line[1]):
					counter += 1

			# text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			text = "{}".format(indexIDs[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			i += 1

	# counting line 그리기
	cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

	# 교차된 차량수 그리기
	cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
	# counter += 1

	# 이미지 파일 저장
	cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

	# video writer가 none인지 확인
	if writer is None:
		# video writer 초기화
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# frame 처리 정보 
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# 비디오 저장하기
	writer.write(frame)

	# frame index 증가
	frameIndex += 1

	if frameIndex >= 4000:
		print("[INFO] cleaning up...")
		writer.release()
		vs.release()
		exit()

# 파일 포인터 해제
print("[INFO] cleaning up...")
writer.release()
vs.release()