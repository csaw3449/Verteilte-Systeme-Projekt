"""
Last modified on 28 March 2019

@author: Roberto Marroquin

@description: Module containing functions related to the DNN models.

@python version: Python 3.4.3
@opencv version: 3.4.2

"""
import cv2
import numpy as np
import functions as fnct


def setup_ddn_detector(detector_name, confThr):
    '''
    Function that setup the architecture of the model, its weights, the classes that it detects and the hardware target
    :param
    :return: model dictionary containing all the information of the architecture.
    '''

    # Load model, configuration, scale, width, height, mean and classes according to the dector
    if detector_name == "YOLOv3_608":
        model, config, scale, width, height, model_classes, mean = set_yolov3_608()
    elif detector_name == "YOLOv3_416":
        model, config, scale, width, height, model_classes, mean = set_yolov3_416()
    elif detector_name == "YOLOv2_608":
        model, config, scale, width, height, model_classes, mean = set_yolov2_608()
    elif detector_name == "YOLOv2_416":
        model, config, scale, width, height, model_classes, mean = set_yolov2_416()
    elif detector_name == "SSD_512":
        model, config, scale, width, height, model_classes, mean = set_ssd_512()
    elif detector_name == "SSD_300":
        model, config, scale, width, height, model_classes, mean = set_ssd_300()
    else:
        print("ERROR: Model is unknown!")
        exit(0)

    # Load names of classes
    classes = None
    if model_classes:
        with open(model_classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

    # Load a network
    network = cv2.dnn.readNet(model, config)

    #backends = (cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_BACKEND_HALIDE, cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,cv2.dnn.DNN_BACKEND_OPENCV)
    # In the PC in runs faster on the CPU!
    #targets = (cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL)

    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Create model dictionary
    detector_keys = ["name", "classes", "model",
                     "confThreshold", "scale", "width", "height", "mean"]
    detector_values = [detector_name, classes,
                       network, confThr, scale, width, height, mean]
    keys_and_values = zip(detector_keys, detector_values)
    detector_dict = {}

    for keys, values in keys_and_values:
        detector_dict[keys] = values
    return detector_dict


def set_yolov3_608():
    """
    Function that sets up YOLOv3_608 detector
    :return: model, config, scale, width, height, mean, classes
    """
    # Load model, configuration, scale, width, height, mean and classes
    model = "resources/models/YOLOv3_608/yolov3-608-coco.weights"
    config = "resources/models/YOLOv3_608/yolov3-coco.cfg"
    scale = 0.00392
    width = 608
    height = 608
    classes = "resources/models/YOLOv3_608/classes_coco.txt"
    mean = (0, 0, 0)

    return model, config, scale, width, height, classes, mean


def set_yolov3_416():
    """
    Function that sets up YOLOv3_416 detector
    :return: model, config, scale, width, height, mean, classes
    """
    # Load model, configuration, scale, width, height, mean and classes
    model = "resources/models/YOLOv3_416/yolov3-416-coco.weights"
    config = "resources/models/YOLOv3_416/yolov3-coco.cfg"
    scale = 0.00392
    width = 416
    height = 416
    classes = "resources/models/YOLOv3_416/classes_coco.txt"
    mean = (0, 0, 0)

    return model, config, scale, width, height, classes, mean


def set_yolov2_608():
    """
    Function that sets up YOLOv2_608 detector
    :return: model, config, scale, width, height, mean, classes
    """
    # Load model, configuration, scale, width, height, mean and classes
    model = "resources/models/YOLOv2_608/yolov2-coco.weights"
    config = "resources/models/YOLOv2_608/yolov2-coco.cfg"
    scale = 0.00392
    width = 608
    height = 608
    classes = "resources/models/YOLOv2_608/classes_coco.txt"
    mean = (0, 0, 0)

    return model, config, scale, width, height, classes, mean


def set_yolov2_416():
    """
    Function that sets up YOLOv2_416 detector
    :return: model, config, scale, width, height, mean, classes
    """
    # Load model, configuration, scale, width, height, mean and classes
    model = "resources/models/YOLOv2_416/yolov2-coco.weights"
    config = "resources/models/YOLOv2_416/yolov2-coco.cfg"
    scale = 0.00392
    width = 416
    height = 416
    classes = "resources/models/YOLOv2_416/classes_coco.txt"
    mean = (0, 0, 0)

    return model, config, scale, width, height, classes, mean


def set_ssd_512():
    """
    Function that sets up YOLOv3_608 detector
    :return: model, config, scale, width, height, mean, classes
    """
    # Load model, configuration, scale, width, height, mean and classes
    model = "resources/models/SSD_512/model.caffemodel"
    config = "resources/models/SSD_512/deploy.prototxt"
    scale = 1
    width = 512
    height = 512
    classes = "resources/models/SSD_512/classes_voc.txt"
    mean = (104, 117, 123)

    return model, config, scale, width, height, classes, mean


def set_ssd_300():
    """
    Function that sets up YOLOv3_608 detector
    :return: model, config, scale, width, height, mean, classes
    """
    # Load model, configuration, scale, width, height, mean and classes
    model = "resources/models/SSD_300/model.caffemodel"
    config = "resources/models/SSD_300/deploy.prototxt"
    scale = 1
    width = 300
    height = 300
    classes = "resources/models/SSD_300/classes_voc.txt"
    mean = (104, 117, 123)

    return model, config, scale, width, height, classes, mean


def dnn_detect_people(frame, detector, label_to_filter):
    """

    """
    # Pre-process image
    # Create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, detector["scale"], (
        detector["width"], detector["height"]), detector["mean"], False, False)

    # Run detector
    detector["model"].setInput(blob)
    # For Faster-RCNN or R-FCN
    if detector["model"].getLayer(0).outputNameToIndex('im_info') != -1:
        frame = cv2.resize(frame, (detector["width"], detector["height"]))
        net.setInput(np.array(
            [detector["width"], detector["height"], 1.6], dtype=np.float32), 'im_info')

    outs = detector["model"].forward(getOutputsNames(detector["model"]))
    # Post-process detections
    detections = postprocess(frame, outs, detector, label_to_filter)
    del blob
    del outs
    return detections


def postprocess(frame, outs, detector, label_to_filter):
    """
    Post process the resulting detection according to the detector used.
    :return
    """
    classes = detector["classes"]
    confThreshold = detector["confThreshold"]
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    layer_names = detector["model"].getLayerNames()
    last_layer_id = detector["model"].getLayerId(layer_names[-1])
    last_layer = detector["model"].getLayer(last_layer_id)
    detections = []
    # Check which type of detection framework we are working with, based on the name of the last layer
    if detector["model"].getLayer(0).outputNameToIndex('im_info') != -1:
        # Faster-RCNN or R-FCN
        detections = postprocess_1(
            outs, confThreshold, classes, label_to_filter, detections)
    elif last_layer.type == 'DetectionOutput':
        # SSD
        detections = postprocess_2(
            outs, confThreshold, classes, label_to_filter, detections, frame_width, frame_height)
    elif last_layer.type == 'Region':
        # YOLO
        detections = postprocess_3(
            outs, confThreshold, classes, label_to_filter, detections, frame_width, frame_height)

    return detections


def postprocess_1(outs, confThreshold, classes, labels_to_filter, detections):
    """
    Used for Faster-RCNN and R-FCN detectors
    """
    # Network produces output blob with a shape 1x1xNx7 where N is a number of
    # detections and an every detection is a vector of values
    # [batchId, class_id, confidence, left, top, right, bottom]
    assert len(outs) == 1
    out = outs[0]
    for detection in out[0, 0]:
        class_id = int(detection[1]) - 1  # Skip background label
        # Is the predicted class label in the set of classes
        # we want to consider?
        if labels_to_filter is not None:
            if not classes[class_id] in labels_to_filter:
                continue
        confidence = detection[2]
        if confidence > confThreshold:
            x = int(detection[3])
            y = int(detection[4])
            w = int(detection[5] - detection[3])
            h = int(detection[6] - detection[4])
            # Check if detection should be considered
            #consider = fnct.check_width_height(w, h)
            # if consider:
            detections.append([
                classes[class_id], float(confidence), x, y, w, h
            ])

    return detections


def postprocess_2(outs, confThreshold, classes, labels_to_filter, detections, frame_width, frame_height):
    """
    Used for SSD detectors
    """
    # Network produces output blob with a shape 1x1xNx7 where N is a number of
    # detections and an every detection is a vector of values
    # [batchId, class_id, confidence, left, top, right, bottom]
    assert len(outs) == 1
    out = outs[0]
    for detection in out[0, 0]:
        class_id = int(detection[1]) - 1  # Skip background label

        # Is the predicted class label in the set of classes
        # we want to consider?
        if labels_to_filter is not None:
            if not classes[class_id] in labels_to_filter:
                continue

        confidence = detection[2]
        if confidence > confThreshold:
            x = int(detection[3] * frame_width)
            y = int(detection[4] * frame_height)
            w = int((detection[5] - detection[3]) * frame_width)
            h = int((detection[6] - detection[4]) * frame_height)
            # Check if detection should be considered
            #consider = fnct.check_width_height(w, h)
            # if consider:
            detections.append([
                classes[class_id], float(confidence), x, y, w, h
            ])
    return detections


def postprocess_3(outs, confThreshold, classes, labels_to_filter, detections, frame_width, frame_height):
    """
    Used for YOLO detectors
    """
    # Network produces output blob with a shape NxC where N is a number of
    # detected objects and C is a number of classes + 4 where the first 4
    # numbers are [center_x, center_y, width, height]
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Is the predicted class label in the set of classes
            # we want to consider?
            if labels_to_filter is not None:
                if not classes[class_id] in labels_to_filter:
                    continue
            if confidence > confThreshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = center_x - width / 2
                top = center_y - height / 2
                # Check if detection should be considered
                #consider = fnct.check_width_height(width, height)
                # if consider:
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    # TODO - NMS could also be use to filter boxes by confidence
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, 0.4)
    for i in indices:
        i = i[0]
        box = boxes[i]

        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        # class_id = int(detection[1]) - 1  # Skip background label
        detections.append([
            classes[class_ids[i]], confidences[i], x, y, w, h])

    return detections


def getOutputsNames(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
