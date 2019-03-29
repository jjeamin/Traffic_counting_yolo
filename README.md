# 차량 counting

비디오에서 차량을 검출하여 counting 해줍니다.

![highway.gif](highway.gif)

It uses:

* [YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv) to detect objects on each of the video frames.

* [SORT](https://github.com/abewley/sort) to track those objects over different frames.

객체를 하나 검출하고 추적한다. 차량의 검출한다음 추적선을 표현하고 정의되어있는 선과 교차할때 counting된다. 

The code on this prototype uses the code structure developed by Adrian Rosebrock for his article [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv).

## 빠른시작

1. git clone
2. [Download yolov3.weights](https://www.dropbox.com/s/99mm7olr1ohtjbq/yolov3.weights?dl=0) 그리고 yolo-coco 폴더에 넣는다.
3. Python 3.7.0 and [OpenCV 3.4.2](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/) 설치하기.
4. 실행:
```
$ python main.py --input input/highway.mp4 --output output/highway.avi --yolo yolo-coco
```
---

# 사람을 웹캠으로 검출하여 실시간 counting 










---

## Citation

### YOLO :

    @article{redmon2016yolo9000,
      title={YOLO9000: Better, Faster, Stronger},
      author={Redmon, Joseph and Farhadi, Ali},
      journal={arXiv preprint arXiv:1612.08242},
      year={2016}
    }

### SORT :

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }
    
