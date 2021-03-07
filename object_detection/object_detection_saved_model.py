#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Saved Model
=====================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


MODEL_FOLDER = 'exported-models'
MODEL_NAME = 'my_model'
PATH_TO_MODEL_DIR = MODEL_FOLDER+'/'+ MODEL_NAME


LABEL_FILENAME = 'annotations/label_map.pbtxt'
PATH_TO_LABELS = LABEL_FILENAME

# Load the model
# ~~~~~~~~~~~~~~
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

IMAGE_SIZE = (12, 8)


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2
from scripts.screen_capture import grabScreen
import time

last_time = time.time()
while True:

    image_np = grabScreen()

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    print("Time: Input {}".format(time.time()-last_time))
    last_time = time.time()
    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    print("Time: Detection {}".format(time.time()-last_time))
    last_time = time.time()
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    print("Time: detetctions array {}".format(time.time()-last_time))
    last_time = time.time()
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)
    print("Time: Vizutils {}".format(time.time()-last_time))
    last_time = time.time()
    cv2.imshow('object_detection',cv2.resize(image_np_with_detections,(400,300)))
    print("Time: Imshow {}".format(time.time()-last_time))
    last_time = time.time()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    cv2.destroyAllWindows()
    break