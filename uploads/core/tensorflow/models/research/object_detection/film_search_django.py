import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import glob
import sqlite3
import cv2

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

print(tf.__version__)

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util


def wordable(video_path, video_id, video_title):
    # 라벨 파일 불러오기
    PATH_TO_LABELS = os.path.join('uploads//core//tensorflow//models//research//object_detection//data',
                                  'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # 데이터베이스 만들기
    BASE_DIR = os.path.abspath('.')
    TARGET_DIR = os.path.join(BASE_DIR, "DB")
    TARGET_FILE = 'test.db'
    TARGET_FILE_FULL_PATH = os.path.join(TARGET_DIR, TARGET_FILE)

    conn = sqlite3.connect('video.db')

    curs = conn.cursor()

    print(",,")

    # form_class = uic.loadUiType("main.ui")[0]

    # form_class2 = uic.loadUiType("frame set.ui")[0]

    global temp
    temp = '1'

    # 슬래쉬를 백슬레쉬로 바꿔주는 함수
    def slash_to_double_backslash(s):
        return s.replace('/', '\\')

    # 카테고리 딕셔너리에서 id를 찾아내는 함수
    def categories_id(n):
        for i in range(0, 80):
            if categories[i]['id'] == n:
                return i

    # 데이터 베이스 인풋 함수
    def input_data(video_num, object_num, time):
        curs.execute('CREATE TABLE IF NOT EXISTS vd(vdnum ,object, time)')
        values = list()
        values.insert(0, (video_num))
        values.insert(1, (object_num))
        values.insert(2, (time))
        # 1. 비디오 번호 2. 물건 번호 3. 시간을 넣음
        curs.execute("INSERT INTO vd VALUES(?, ?, ?)", values)

        conn.commit()

        a = curs.execute("SELECT * FROM vd")

        rows = a.fetchall()

        row_counter = 0

        return

        # 데이터 베이스에서 검색하는 함수

    def do_search_object():
        global curs
        v = self.lineEdit.text()

        a = curs.execute("SELECT * FROM vd WHERE object == '%s'" % (v))

        rows = a.fetchall()

        row_counter = 0

        # 출력
        for row in rows:
            l = row[0]
            m = row[1]
            n = row[2]

            self.textEdit.insertPlainText("비디오 번호 : %s  물체 번호 : %s  등장 시간 : %sm %.3lfs\n" % (l, m, int(n / 60), n % 60))
            row_counter += 1

        self.textEdit.insertPlainText("=========================================================\n")

        row_counter += 1

    # 영상 폴더 경로 지정함수
    def find_folder_path():
        global file_list
        global foldername
        file_list = list()

        foldername = QFileDialog.getExistingDirectory()
        self.label_5.setText(foldername)

        if foldername != '':
            for file in os.listdir(foldername):
                if file.endswith(".mp4"):
                    file_list.append(file)

    def db_init():

        curs.execute('DROP TABLE IF EXISTS vd')
        curs.execute('CREATE TABLE vd(vdnum ,object, time)')

    def get_filename(filename):

        a = slash_to_double_backslash(filename)
        b = a
        while True:
            a = b
            try:
                b = a.split(sep='\\', maxsplit=1)[
                    1]  # 경로가 저장된 문자열에서 \\가 나오기 전까지 지움 ex) ssd_mobilenet_v1_coco_2018_01_28.tar.gz 만 남게됨
            except:
                if a == b: break
            print(b)
        filename = b
        return filename

        # 동영상 분석 함수

    def do_scan():
        global frame_num
        frame_num = 3
        global filename
        filename = "ssd_mobilenet_v1_coco_2018_01_28.tar.gz"
        global foldername
        global show_analysis_video
        # global video_path
        # video_path = "/workspace/builderschallenge/simple-file-upload-master/media/1.mp4"

        MODEL_NAME = filename  # 선택한 모델파일을 불러옴
        MODEL_FILE = MODEL_NAME  # + '.tar.gz'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME.rstrip('.tar.gz') + '/frozen_inference_graph.pb'

        opener = urllib.request.URLopener()
        # DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        # opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        print(frame_num)

        cnt = 0

        # 동영상 갯수만큼 반복

        cap = cv2.VideoCapture(video_path)

        # 총 프레임수와 초당 프레임 계산
        print(cap.get(cv2.CAP_PROP_FPS))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                ret = True

                # ret,image_np=cap.read()
                cnt2 = 0

                objects = list()
                objects_time = list()

                while (cap.isOpened()):

                    cnt2 += 1
                    ret, image_np = cap.read()  # 프레임 불러오기

                    if cap.isOpened() == 0: break

                    if cnt2 % int(frame_num) == 0:  # 분석해야할 프레임이면 분석
                        try:
                            # Definite input and output Tensors for detection_graph
                            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                            # Each box represents a part of the image where a particular object was detected.
                            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                            # Each score represent how level of confidence for each of the objects.
                            # Score is shown on the result image, together with the class label.
                            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image_np, axis=0)
                            # Actual detection.
                            (boxes, scores, classes, num) = sess.run(
                                [detection_boxes, detection_scores, detection_classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})

                            for i in range(0, 30):
                                if scores[0][i] > 0.80:  # 한 프레임에서 인식된 물체중에서 일치도 상위 30개중 일치도가 60퍼 이상이면 저장
                                    print("a")
                                    print(int(classes[0][i]))
                                    print(categories_id(int(classes[0][i])))
                                    print(categories[categories_id(int(classes[0][i]))]['name'])

                                    objects.insert(cnt, categories[categories_id(int(classes[0][i]))]['name'])
                                    objects_time.insert(cnt, cnt2 / fps)
                                    flag = 0
                                    cnt += 1

                                    # Visualization of the results of a detection.
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=3)

                            # cv2.imshow('live_detection', image_np)  # 분석 프레임 보여주기

                            # 동영상이 끝날때
                        except:
                            cv2.destroyAllWindows()
                            print("except")
                            cap.release()

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        cap.release()
                        break

                # cv2.destroyAllWindows()
                # print("ghjk")
        cap.release()

        # filename = get_filename(video_path)
        # 데이터 베이스에 저장
        for i in range(1, cnt):
            # print("789")
            input_data(video_id, objects[i], video_title)

        cnt = 0
        print("video over")

    do_scan()
    return

# wordable('/workspace/builderschallenge/simple-file-upload-master/media/1.mp4')
