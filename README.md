# EEC174 AY Project A1: YOLO Object Detection and MOT

Due Date: November 4, 2022

The focus of this lab is to learn how to apply YOLO on images & videos. In Phase 1 you will learn to load the YOLOv3 model into OpenCV, create blobs for OpenCV, detect objects and drawing bounding boxes, labels onto images. In Phase 2 you will use open-source MOT solutions with YOLO to perform object tracking on video inputs. Please read all the instructions carefully.

## Environment Setup

You can utilize the Lab 3 container (built from image ```eec174lab1-2```) or create an new container. Please install all additional dependancies that are in `requirements.txt` (with either conda or pip).

All required files are provided except the pretrained weights (as they were too large to upload on github). You will need to download the weights file in the `yolo_files` folder by using the following command:

```
wget https://pjreddie.com/media/files/yolov3.weights
```

(Note: When submitting do not upload the weights!)

## YOLO Tutorial

The [YOLO_OpenCV_Tutorial.ipynb](src/YOLO_OpenCV_Tutorial.ipynb) contains the full inference pipeline and code to draw bounding boxes onto a given image using OpenCV & YOLOv3. 
The given code provides a way to feed an image and draw (on the same image) bounding boxes and confidence score. The tutorial is adapted from the [OpenCV tutorial](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html). You will have to modify the code provided to complete Phase 1 & 2.

## Phase 1: YOLO Object Detection on Images 

Your task is to complete the [yolo_img_detector.py](src/yolo_img_detector.py) to perform object detection (with YOLOv3). This script must be able to take in path to image/images/folder of images and perform YOLO to report the following main tasks:

- The average inference time (across all input images specified)
- The total number of classes detected
- A breakdown for each class by class name and corresponding number of those classes found
- Store all specified images with bounding boxes, class names, and confidence scores drawn in `out_imgs`
- The output drawn/annotated images must be named with a '_out' attached in following style: `img_out.jpg` (for given `input img.jpg`)

You must adapt code from the tutorial to do this. The given tutorial does not currently add the class names, only shows the confidence and bounding boxes. You must read the `coco.names` file and also include the correct class names by index. All test input images can be found in `imgs/`

Further requirements/functionalities:

- You must be able to pass in paths to single images, multiple selected images, or a folder with images in them. The script must then work on all the images.
- You cannot hard code the paths to images, weights or other files.
- The script must have an interface with flags for each argument and a `-h` help option to show functionality

How you decide to implement your script and read CLI arguments is up to you. You are required to take the following inputs:

- YOLO Weights file
- YOLO config file
- Labels (coco.names)
- Path to image/images: you must be able to pass single image, a list of images, or a directory with images (.jpg images)

### Phase 1: Example Use 

There is some amount of freedom how you wish to present information as long as it can all be presented. For instance, if you are passing the given ```sample.jpg``` your output ***may*** look like the following:

(Your script should have a way to show all the following information - not necessarily together):

```
Average Inference Time: 2.61351 seconds
Total Number of Classes Detected: 1

Total Detection Breakdown
Car: 1

Per Image Breakdown
sample.jpg => Car: 1
```

You can have different flags to show different parts of the information and display only parts of the information as specified by the script. A good script would seperate parts by selected flags. 

For instance, a possible design idea can be to  have a ```-inf``` flag to show only the inference time, or ```-classes_all``` to show total number of classes detected:

```
yolo_img_detector.py <inputs> -inf
```
and the possible output can be:
```
Average Inference Time: 2.61351 seconds
```
Another example:
```
yolo_img_detector.py <inputs> -inf -classes_all
```
and the possible output can be:
```
Average Inference Time: 2.61351 seconds
Total Number of Objects/Classes Detected: 1
```

You must save all annotted images in `/out_imgs` with correct name. Here is an example of an input and output image:

input: ```sample.jpg```

![sample.jpg](imgs/sample.jpg)

output: ```sample_out.jpg```

![sample_out.jpg](out_imgs/sample_out.jpg)

***Reminder***: The examples shown only pass one image but your script should allow multiple. This affects the results. 

For instance if you are to pass ```sample.jpg``` and ```img/elephants.jpg``` to your script, the results should look like this:

```
Average Inference Time: 2.12312 seconds
Total Number of Objects/Classes Detected: 3

Total Detection Breakdown
Car: 1
Elephant: 2

Per Image Breakdown
sample.jpg => Car: 1
elephants.jpg => Elephant: 2
```

## Phase 2: YOLO Object detection & MOT on videos

For this phase, you will be working on [yolo_counter.py](src/yolo_counter.py) which takes in a video and counts the number of people (using Multi-Object Tracking code). You are not expected to code the MOT algorithm, instead we will utilize an open-source code called SORT. The required code is already in your repository ([sort.py](src/sort.py)), so please do not clone the SORT repo and do not modify `sort.py`. You will need to go through the [sort.py](src/sort.py) and [sort.md](sort.md) understand how to call the MOT tracker.
The input video is in `mot_vid/` folder, called `mot_vid/MOTS20-09-raw.mp4`. 

The required inputs to ```yolo_counter.py``` will be same as Phase 1, except we take an input video instead of images:
- YOLO Weights file
- YOLO config file
- Labels (coco.names)
- Path to input video (.mp4)

The output ***must*** be the annotated video saved as `mot_vid/MOTS20-09-result.mp4`. As your program runs, please also print out the frame number and processing time taken per frame (in s).

The expected result video should look like this (not full video shown):
![](https://github.com/EEC193AB-master/kartikp7LabA1/blob/main/mot_vid/MOTS20-09-output.gif)

### Phase 2 Suggested Steps

#### 1. YOLO on video input
Start by porting your code from phase 1 to perform on video input and output/save a video with bounding boxes ([OpenCV: Getting Started with Videos](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html))

#### 2. Detect only People (person class)
You will need to modify the code to only draw/annotate bounding boxes for people

#### 3. Use MOT tracker to track & count
Finally, utilize the SORT tracker to assign IDs to bounding boxes to track and count (see [sort.md](sort.md) and [sort.py](src/sort.py) on how to use SORT).

## Evaluation

### Grade Breakdown

- Program correctness: 90%
- Program usability & design: 10%

This is an individual assignment, so you must work alone. Please avoid excessive collaboratation. Please do not use code from SORT repo, only the [sort.py](src/sort.py) provided in your repository.

#### Phase 1 correctness (50%)
Your script will be run on a new set of images and directory of images. Full functionality will be tested to make sure the correct information is reported from your script. You can easily identify and test your output by observing the given images. You can visually check the images and count the bounding boxes, check the class names.

#### Phase 2 correctness (40%)
The TA program detects total 107 people at the end of the video. The sample video also provides a working demo. During Lab walkthrough, the TA will go over the output video and what to expect.
The colors, location of counter can vary as long as all information is correct. you can test and debug your code by saving some initial frames.

#### Program Usability & Design (10%)

Please make sure your programs are well commented and readable. Make sure to adress invalid CLI inputs or missing inputs. If the programs are used incorrectly, your programs should exit safetly and output the correct use. Have functions wherever necessary for readability.

## Submission

Due Date: November 4, 2022

Graded files:
- ```yolo_img_detector.py```
- ```yolo_counter.py```
- Phase 2 output video

Commits up to a week late will incur a -10% late penalty.

## Credits

Kartik Patwari, Jeff Lai, and Chen-Nee Chuah

## References

[OpenCV: YOLO](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)

[OpenCV: Getting Started with Videos](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html)

[OpenCV: YOLO on videos](https://docs.opencv.org/4.x/da/d9d/tutorial_dnn_yolo.html)

[SORT: MOT tracker](https://github.com/abewley/sort)
