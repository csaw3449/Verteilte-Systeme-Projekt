###General notes
The scripts were tested in Ubuntu 18.04 and Linux Mint 17.3

All the scripts should be executed from the "scripts/" directory.

###Installation
To install the python dependencies use the command

    `pip3 install -r requirements.txt`

#####Remark    
Cuda should be install to execute opencv's dnn model. If is not installed, the cuda package `cuda-repo-ubuntu1804_10.0.130-1_amd64.deb` can be found in the `scripts/resources/` directory.

The steps to install it are:

   `sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb`
   `sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub`
   `sudo apt-get update`
   `sudo apt-get install cuda`

Preferably, the opencv-python should be re-installed.

###Functionalities of each script:
+ `automatic_people_detector.py` --> script to generate the automatic annotations
+ `detections_visualizer.py` --> script to visualize detections (manual or automatic)
+ `json_to_txt.py` --> script to extract the detections in the JSON file and store them in separate text files.
					  The evaluation is then performed using the code found at: https://github.com/rafaelpadilla/Object-Detection-Metrics

###Usage examples
#####automatic_people_detector.py
Example to run the YOLOv3 detector on the video3_1, using as feature descriptor Hue-Saturation (HS) histogram and background-suppresion (BS):
  `python3 automatic_people_detector.py --input ../video_sets/set_3/video3_1.avi --show 1 --backgroundSupp 1 --feature HS --detector YOLOv3_608`

######Notes:
+ there are 3 types of detectors that can be used: HOG_SVM, SSD_512 or YOLOv3_608.
+ there are 2 types of feature descriptor that can be used: HS or HS_RGB histograms.
+ the feature descriptor can be extracted from the complete detection or just from a localized region (`--backgroundSupp 1`)
+ the resulting annotations will be stored in:
 `scripts/automatic_annotations/<detector_name>/<feature_descriptor_name>/<set_number>/`

#####detections_visualizer.py
Example to visualize the manual annotations of video4_4:
  `python3 detections_visualizer.py --file ../manual_annotations/people_detection/set_4/video4_4.json --id 1`

Example to visualize the automatic annotations of YOLOv3_608 of video4_4:
  `python3 detections_visualizer.py --file ../automatic_annotations/YOLOv3_608/HS_BS/set_4/video4_4.json`

######Notes:
+ the `--id` flag indicates if the ID information is present for each detections, therefore it should be set to true (1) when using manual annotations.

#####json_to_txt.py
Example to extract in txt files the manual annotations of set 1:
  `python3 json_to_txt.py -dir ../manual_annotations/people_detection/set_1 -gt 1`

Example to extract in txt files the automatic annotations using HOG_SVM, of set 1:
  `python3 json_to_txt.py -dir ../automatic_annotations/HOG_SVM/HS_BS/set_1 -gt 1`  

######Notes  
+ the `-gt` flag indicates if the annotations is ground truth (manual) or not, thus it should be set to true (1) when using manual annotations.
+ the resulting txt files will be stored in:
  `scripts/manual_annotations_txt/`

  or in:

  `scripts/automatic_annotations_txt/`
