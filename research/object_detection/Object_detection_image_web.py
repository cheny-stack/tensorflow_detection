######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import datetime
import json
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph1'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training1','labelmap.pbtxt')



# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# tf.device('/gpu:3')
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

# Load the Tensorflow model into memory.
sess =None
detection_graph=None
image_tensor=None
detection_boxes=None
detection_scores=None
detection_classes=None
num_detections=None
def init_detect():
    global sess
    global detection_graph
    global image_tensor
    global detection_boxes
    global detection_scores
    global detection_classes
    global num_detections
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
id_name_str=u'[{"id":1,"name":"qg","real_name":"秦刚"},{"id":2,"name":"xl","real_name":"许玲"},{"id":3,"name":"cb","real_name":"蔡晓东"},{"id":4,"name":"gg","real_name":"黄朝光"},{"id":5,"name":"fg","real_name":"梁发记"},{"id":6,"name":"mg","real_name":"陈明瑶"},{"id":7,"name":"ly","real_name":"李燕"},{"id":8,"name":"other","real_name":"其他"}]'
id_name_map_list= json.loads(json.loads(json.dumps(id_name_str),encoding="UTF-8"))
id_name_map={}
for p in id_name_map_list:
    name=p['name']
    id_name_map[name]=p['real_name']


from flask import Flask,request,make_response

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def buildHtml(resList):
    base= '''
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8">
    <title>识别结果</title>
    <style>
        img{{
            max-height: 150px;
            max-width: 150px;
        }}
        .search-res{{
            display: inline-block;
            width: 150px;
            height: 300px;
        }}
    </style>
    </head>
    <body>
    {res_div}
    </body>
    </html>
    '''
    div_str=""
    div="<div class='search-res'><h4>姓名：{name}</h4><h4>相似度：{sim}</h4><img src='{url}' alt='图片'></div>"
    for res in resList:
        url="http://localhost:5000/api/show/"+res["file_name"]
        div_str += div.format(name=res['name'],sim=res["sim"],url=url)
    return base.format(res_div=div_str)


@app.route('/api/face_detection', methods=['POST'])
def face_detection():
    res_list=[]
    fs = request.files.getlist('file[]')
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    for f in fs :
        res = {'code': '0'}
        date = datetime.datetime.now()
        detester = date.strftime('%Y%m%d%H%M%S')
        file_extension = os.path.splitext(f.filename)[1]
        upload_path = os.path.join(basepath, 'static', 'uploads')  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        if not os.path.isdir(upload_path):
            os.makedirs(upload_path)
        file_name = detester + file_extension
        upload_path = os.path.join(upload_path, file_name)
        print("upload_path:" + upload_path)
        f.save(upload_path)
        res['file_name'] = file_name
        res['upload_path'] = upload_path
        # file_path = request.form.get('file_path')
        print(str(upload_path))
        PATH_TO_IMAGE = upload_path
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        name_sim = vis_util.visualize_boxes_and_labels_on_image_array_cy(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            max_boxes_to_draw=1,
            min_score_thresh=0.70)
        # cv2.imwrite(path_save+file, image)
        name = None
        sim=None
        if name_sim != None:
            name_sim_array = str(name_sim).split(":")
            sim = str(name_sim_array[1])
            sim = float(str(sim).strip("%")) / 100

            if name_sim_array[0] in id_name_map and sim > 0.8:
                name = id_name_map[str(name_sim_array[0])]
            else:
                name="其他"

        print(name)
        res['name'] = name
        res['sim'] = sim
        res_list.append(res)
    # return json.dumps(res_list,ensure_ascii=False)
    return buildHtml(res_list)

# show photo
@app.route('/api/show/<string:filename>', methods=['GET'])
def show_photo(filename):
    basepath = os.path.dirname(__file__)
    file_dir = os.path.join(basepath, 'static','uploads')
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            file_path=os.path.join(file_dir, filename)
            image_data = open(file_path, "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass

if __name__ == '__main__':
    init_detect()
    app.run(host='0.0.0.0', port=5000)


