#The WiseNET dataset
The WiseNET dataset provides multi-camera multi-space video sets, along with manual and automatic people detection/tracking annotations and the complete contextual information of the environment where the network was deployed.

###video_sets/ :
The dataset regroups 11 video sets (composed of 62 single videos) recorded using 6 indoor cameras deployed on multiple spaces. The video sets represent more than one hour of video footage, include 69 people tracks and captured different human actions such as walking around, standing/sitting, motionless, entering/leaving a space and group merging/splitting.

The position of each camera can be seen in the figure `network_environment/topology.eps`

###manual_annotations/ :
Each video has been manually annotated with people detection and tracking meta-information.
The people detection annotations, were performed on each single video in the sets. While the people tracking annotations, were performed on the complete video sets.

Both types of annotations were stored in JSON files. Moreover, for the people tracking a space-time graph is also provided.

###automatic_annotations/ :
Automatic people detection annotations are also provided for each single video in the sets. The annotations were obtained by using the HOG people detector and two state-of-the-art CNN-based object detectors: SSD and YOLOv3.

The automatic annotations were obtained using a linux server with 4 Core CPU@3GHz and 6GB of RAM.

###network_environment/ :
The Industry Foundation Classes (IFC) file (`I3M.ifc`) that represents the environment's Building Information Modeling (BIM) data is also provided. Also, it includes `camera_calibration` files for each camera node at different resolutions. The calibration files include the location of each camera node, as well as the position of the doors of interest it observes.

###scripts/ :
We are also providing python scripts to:
+ visualize the manual and automatic people annotations --> `detections_visualizer.py`
+ generate the automatic people annotations --> `automatic_people_detector.py`
+ transform the people annotations from JSON to text, in order to use them with external evaluation
frameworks --> `json_to_txt.py`

All scripts should be executed from the `scripts/` directory.

A `requirements.py` file is provided to install the necessary dependencies.

The `resources/` directory includes the network structure and weights of the CNN-based detectors.

###License :
Please refer to the `LICENSE.txt` file for more information concerning the license of the dataset.
