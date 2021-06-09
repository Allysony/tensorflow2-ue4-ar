# Import data
# from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

import numpy as np
import cv2
import math

from config import config
from mathHelperFunctions import normalize, sigmoid
from poseHelperFunctions import get_keypoint_positions, calc_offsets


tf_version = "TENSORFLOW VERSION: " + tf.__version__
ue.log(tf_version)

class PoseEstimate(TFPluginAPI):

    #expected api: setup your model for your use cases
    def onSetup(self):
        #setup or load your model and pass it into stored
        self.interpreter = tf.lite.Interpreter(model_path=config['tflite_model'])
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()


        #Usually store session, graph, and model if using keras
        # self.sess = tf.InteractiveSession()
        # self.graph = tf.get_default_graph()

    #expected api: storedModel and session, json inputs
    def onJsonInput(self, jsonInput):
        #e.g. our json input could be a pixel array
        #pixelarray = jsonInput['pixels']

        #run input on your graph
        #e.g. sess.run(model['y'], feed_dict)
        # where y is your result graph and feed_dict is {x:[input]}

        #...

        #you can also call an event e.g.
        #callEvent('myEvent', 'myData')

        #return a json you will parse e.g. a prediction
        result = {}
        #result['prediction'] = -1

        frame = jsonInput['frame']     # Input Image (640, 480, 3)

        new_img = cv2.resize(frame, (config['IN_IMAGEW'], config['IN_IMAGEH']))
        new_img = normalize(new_img)
        new_img_tf = np.expand_dims(new_img.astype(np.float32), axis=0)

        # Predict
        heatmaps = interpreter.get_tensor(output_details[0]['index'])  # Heatmaps (1, 9, 9, 17)
        offsets  = interpreter.get_tensor(output_details[1]['index'])  # Offsets (1, 9, 9, 34)

        h = heatmaps.shape[1]
        w = heatmaps.shape[2]
        num_kps = heatmaps.shape[3]

        kps_pos = get_keypoint_positions(heatmaps, num_kps, h, w)
        x_coords, y_coords, conf_scores = calc_offsets(heatmaps, kps_pos, offsets, num_kps, h, w)

        final_kps = []
        for idx in range(config['num_joints']):
            x = x_coords[idx]
            y = y_coords[idx]
            if conf_scores[idx] > config['min_confidence']:
                final_kps.append((x, y))
            else:
                final_kps.append((None, None))

        for idx in range(len(final_kps)):
            x, y = final_kps[idx]
            cv2.line(frame, (x, y), (x, y), (215, 150, 225), config['circle_radius'])

        for edge in config['edges']:
            x0, y0 = final_kps[edge[0]]
            x1, y1 = final_kps[edge[1]]

        if x0 and y0 and x1 and y1:
            cv2.line(frame, (x0, y0), (x1, y1), (215, 150, 225), config['edge_radius'])


        result['predictions'] = frame
        ue.log(frame)

        return result

    #optional api: no params forwarded for training? TBC
    def onBeginTraining(self):
        #train here

        #...

        #inside your training loop check if we should stop early
        #if(this.shouldStop):
        #	break
        pass

    #optional api: use if you need some things to happen if we get stopped
    def onStopTraining(self):
        #you should be listening to this.shouldstop, but you can also receive this call
        pass

#required function to get our api
def getApi():
    #return CLASSNAME.getInstance()

    return PoseEstimate.getInstance()
