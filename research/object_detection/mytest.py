import os

path="/home/omnisky/chenyun/workspase/models-r1.5/research/object_detection/images/test_detection/"
path="/home/omnisky/chenyun/workspase/models-r1.5/research/object_detection/images/test_detection/res/"
file_list=os.listdir(path)
for file in file_list:
    print(path+file)