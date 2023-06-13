import easygui as eg
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import sys
import requests
import json
import base64
import os
import logging
import speech_recognition as sr
from queue import PriorityQueue
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def face_recognition(paths):
    judge = True
    faceCascade = cv2.CascadeClassifier('face_origin_datacasades/cascades/haarcascade_frontalface_default.xml')
    paths = paths.replace('\\','/')
    img =cv2.imread(paths)
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor= 1.1,
    minNeighbors=8,
    minSize=(55, 55),
    flags=cv2.CASCADE_SCALE_IMAGE
    )


    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


    finalimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.figure(figsize=(20, 20))
    plt.imshow(finalimg)
    plt.axis("off")
    plt.show()


    print(len(faces), "faces detected!")

    if (len(faces)<=5):
        print("no overload")
    else:
        print("Overcrowded!")
    return judge

def select_path():
    paths = eg.fileopenbox()
    return paths


def select_module():
    module = eg.buttonbox(msg='                 Please choose what you need.', title='Go for it - CabeCar！', choices=('Image identification', 'Speech Recognition', 'Traffic light detection','Route planning','None'),images='car3.gif', default_choice=('Button[4]'),cancel_choice=None,callback=None, run=True)

    return module

def camera_face_recognition():
    cap = cv2.VideoCapture(0)
    eg.msgbox('                          Press q to exit the camera' + '\n'+'\n' +'                          Press Yes to open the camera', title='Mask recognition',ok_button='Yes')

    while (True):

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        xmlfile = r'face_origin_datacasades/cascades/haarcascade_frontalface_default.xml'
        xmlfile_1 = r'face_origin_datacasades/cascades/cascade.xml'


        face_cascade = cv2.CascadeClassifier(xmlfile)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 255, 0), 1)

        face_cascade01 = cv2.CascadeClassifier(xmlfile_1)
        faces01 = face_cascade01.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(5, 5),
        )


        for (x, y, w, h) in faces01:
            cv2.rectangle(frame, (x, y), (x + w, y + w), (0, 0, 255), 5)

        cv2.imshow("Camera detection", frame)
        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break


    cap.release()

    cv2.destroyAllWindows()


from queue import PriorityQueue
def Dijkstra():
    g = Graph(26)
    g.add_edge(0, 1, 3)
    g.add_edge(0, 2, 4)
    g.add_edge(1, 3, 7)
    g.add_edge(1, 4, 6)
    g.add_edge(1, 2, 5)
    g.add_edge(2, 5, 4)
    g.add_edge(4, 5, 3)
    g.add_edge(3, 10, 2)
    g.add_edge(3, 9, 7)
    g.add_edge(4, 8, 9)
    g.add_edge(5, 8, 8)
    g.add_edge(5, 7, 2)
    g.add_edge(5, 6, 8)
    g.add_edge(6, 11, 11)
    g.add_edge(6, 12, 12)
    g.add_edge(11, 12, 7)
    g.add_edge(7, 12, 7)
    g.add_edge(7, 8, 7)
    g.add_edge(8, 12, 8)
    g.add_edge(8, 9, 6)
    g.add_edge(9, 10, 7)
    g.add_edge(9, 14, 2)
    g.add_edge(8, 13, 8)
    g.add_edge(8, 17, 11)
    g.add_edge(12, 17, 4)
    g.add_edge(12, 23, 5)
    g.add_edge(17, 23, 5)
    g.add_edge(17, 24, 7)
    g.add_edge(13, 17, 2)
    g.add_edge(13, 18, 7)
    g.add_edge(13, 14, 2)
    g.add_edge(14, 15, 2)
    g.add_edge(10, 15, 7)
    g.add_edge(10, 16, 5)
    g.add_edge(15, 16, 6)
    g.add_edge(16, 20, 4)
    g.add_edge(15, 20, 3)
    g.add_edge(15, 19, 7)
    g.add_edge(13, 19, 2)
    g.add_edge(18, 19, 4)
    g.add_edge(18, 24, 4)
    g.add_edge(21, 24, 2)
    g.add_edge(24, 25, 7)
    g.add_edge(21, 25, 1)
    g.add_edge(20, 21, 5)
    g.add_edge(19, 20, 2)
    g.add_edge(19, 21, 5)
    g.add_edge(16, 22, 4)
    g.add_edge(21, 22, 3)
    g.add_edge(3, 4, 3)
    dic = {'Angola': 0, 'Bahrian': 1, 'Congo': 2, 'Drige': 3, 'Enheri': 4, 'Fqeso': 5, 'Gers': 6, 'Hsaid': 7,
           'Icsaf': 8,
           'Jcof': 9, 'Kmsd': 10, 'Lcxu': 11, 'Msdad': 12, 'Nsdi': 13, 'Ocy': 14, 'Pxcn': 15, 'Qoc': 16, 'Rkcsd': 17,
           'Skck': 18,
           'Txkc': 19, 'Unc': 20, 'Vmxc': 21, 'Wkc': 22, 'Xsiad': 23, 'Ysoad': 24, 'Zsda': 25}
    dic1 = {0: 'Angola', 1: 'Bahrian', 2: 'Congo', 3: 'Drige', 4: 'Enheri', 5: 'Fqeso', 6: 'Gers', 7: 'Hsaid',
            8: 'Icsaf',
            9: 'Jcof', 10: 'Kmsd', 11: 'Lcxu', 12: 'Msdad', 13: 'Nsdi', 14: 'Ocy', 15: 'Pxcn', 16: 'Qoc', 17: 'Rkcsd',
            18: 'Skck',
            19: 'Txkc', 20: 'Unc', 21: 'Vmxc', 22: 'Wkc', 23: 'Xsiad', 24: 'Ysoad', 25: 'Zsda'}
    start_point = ""
    end_point = ""
    while True:
        start_point = input("The start place is :")
        end_point = input("The end place is :")
        a =dic[start_point]
        b =dic[end_point]
        break
    D, previousVertex = g.dijkstra(a)

    path = []
    cheapest_path = []
    key=b


    while True:
        if key == a:
            path.append(a)
            break

        else:
            path.append(key)
            key = previousVertex[key]


    for point in path[::-1]:

        cheapest_path.append(dic1[point])

    cheapest_path = "->".join(cheapest_path)

    print(f"Distance from {dic1[int(a)]} to {dic1[int(b)]} is {D[b]} km,"
          f"The optimal route is {cheapest_path}")



class Graph:

    def __init__(self, num_of_vertices):
        self.vertices = num_of_vertices

        self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]

        self.visited = []

    def add_edge(self, u, v, weight):

        self.edges[u][v] = weight

        self.edges[v][u] = weight

    def dijkstra(self, start_vertex):

        D = {v: float('inf') for v in range(self.vertices)}

        D[start_vertex] = 0

        pq = PriorityQueue()

        pq.put((0, start_vertex))

        previousVertex = {}

        while not pq.empty():

            (dist, current_vertex) = pq.get()

            self.visited.append(current_vertex)

            for neighbor in range(self.vertices):

                if self.edges[current_vertex][neighbor] != -1:
                    distance = self.edges[current_vertex][neighbor]

                    if neighbor not in self.visited:
                        old_cost = D[neighbor]
                        new_cost = D[current_vertex] + distance

                        if new_cost < old_cost:

                            pq.put((new_cost, neighbor))
                            D[neighbor] = new_cost
                            previousVertex[neighbor] = current_vertex

        return D, previousVertex




def Dijkstra1():
    dic = {'Angola': 0, 'Bahrian': 1, 'Congo': 2, 'Drige': 3, 'Enheri': 4, 'Fqeso': 5, 'Gers': 6, 'Hsaid': 7,
           'Icsaf': 8,
           'Jcof': 9, 'Kmsd': 10, 'Lcxu': 11, 'Msdad': 12, 'Nsdi': 13, 'Ocy': 14, 'Pxcn': 15, 'Qoc': 16, 'Rkcsd': 17,
           'Skck': 18,
           'Txkc': 19, 'Unc': 20, 'Vmxc': 21, 'Wkc': 22, 'Xsiad': 23, 'Ysoad': 24, 'Zsda': 25}
    dic1 = {0: 'Angola', 1: 'Bahrian', 2: 'Congo', 3: 'Drige', 4: 'Enheri', 5: 'Fqeso', 6: 'Gers', 7: 'Hsaid',
            8: 'Icsaf',
            9: 'Jcof', 10: 'Kmsd', 11: 'Lcxu', 12: 'Msdad', 13: 'Nsdi', 14: 'Ocy', 15: 'Pxcn', 16: 'Qoc', 17: 'Rkcsd',
            18: 'Skck',
            19: 'Txkc', 20: 'Unc', 21: 'Vmxc', 22: 'Wkc', 23: 'Xsiad', 24: 'Ysoad', 25: 'Zsda'}
    while True:
        start_point1 = input("The start place is :")
        end_point1 = input("The place you should pass is :")
        end_point2 = input("The end place is:")
        a = dic[start_point1]
        b = dic[end_point1]
        c = dic[end_point2]
        break
    g = Graph(26)
    g.add_edge(0, 1, 3)
    g.add_edge(0, 2, 4)
    g.add_edge(1, 3, 7)
    g.add_edge(1, 4, 6)
    g.add_edge(1, 2, 5)
    g.add_edge(2, 5, 4)
    g.add_edge(4, 5, 3)
    g.add_edge(3, 10, 2)
    g.add_edge(3, 9, 7)
    g.add_edge(4, 8, 9)
    g.add_edge(5, 8, 8)
    g.add_edge(5, 7, 2)
    g.add_edge(5, 6, 8)
    g.add_edge(6, 11, 11)
    g.add_edge(6, 12, 12)
    g.add_edge(11, 12, 7)
    g.add_edge(7, 12, 7)
    g.add_edge(7, 8, 7)
    g.add_edge(8, 12, 8)
    g.add_edge(8, 9, 6)
    g.add_edge(9, 10, 7)
    g.add_edge(9, 14, 2)
    g.add_edge(8, 13, 8)
    g.add_edge(8, 17, 11)
    g.add_edge(12, 17, 4)
    g.add_edge(12, 23, 5)
    g.add_edge(17, 23, 5)
    g.add_edge(17, 24, 7)
    g.add_edge(13, 17, 2)
    g.add_edge(13, 18, 7)
    g.add_edge(13, 14, 2)
    g.add_edge(14, 15, 2)
    g.add_edge(10, 15, 7)
    g.add_edge(10, 16, 5)
    g.add_edge(15, 16, 6)
    g.add_edge(16, 20, 4)
    g.add_edge(15, 20, 3)
    g.add_edge(15, 19, 7)
    g.add_edge(13, 19, 2)
    g.add_edge(18, 19, 4)
    g.add_edge(18, 24, 4)
    g.add_edge(21, 24, 2)
    g.add_edge(24, 25, 7)
    g.add_edge(21, 25, 1)
    g.add_edge(20, 21, 5)
    g.add_edge(19, 20, 2)
    g.add_edge(19, 21, 5)
    g.add_edge(16, 22, 4)
    g.add_edge(21, 22, 3)
    g.add_edge(3 , 4, 3)

    D, previousVertex = g.dijkstra(b)
    path = []
    cheapest_path = []
    key = c
    while True:
        if key == b:
            path.append(b)
            break

        else:
            path.append(key)
            key = previousVertex[key]
    for point in path[::-1]:
        cheapest_path.append(dic1[point])

    cheapest_path = "->".join(cheapest_path)
    g = Graph1(26)
    g.add_edge(0, 1, 3)
    g.add_edge(0, 2, 4)
    g.add_edge(1, 3, 7)
    g.add_edge(1, 4, 6)
    g.add_edge(1, 2, 5)
    g.add_edge(2, 5, 4)
    g.add_edge(4, 5, 3)
    g.add_edge(3, 10, 2)
    g.add_edge(3, 9, 7)
    g.add_edge(4, 8, 9)
    g.add_edge(5, 8, 8)
    g.add_edge(5, 7, 2)
    g.add_edge(5, 6, 8)
    g.add_edge(6, 11, 11)
    g.add_edge(6, 12, 12)
    g.add_edge(11, 12, 7)
    g.add_edge(7, 12, 7)
    g.add_edge(7, 8, 7)
    g.add_edge(8, 12, 8)
    g.add_edge(8, 9, 6)
    g.add_edge(9, 10, 7)
    g.add_edge(9, 14, 2)
    g.add_edge(8, 13, 8)
    g.add_edge(8, 17, 11)
    g.add_edge(12, 17, 4)
    g.add_edge(12, 23, 5)
    g.add_edge(17, 23, 5)
    g.add_edge(17, 24, 7)
    g.add_edge(13, 17, 2)
    g.add_edge(13, 18, 7)
    g.add_edge(13, 14, 2)
    g.add_edge(14, 15, 2)
    g.add_edge(10, 15, 7)
    g.add_edge(10, 16, 5)
    g.add_edge(15, 16, 6)
    g.add_edge(16, 20, 4)
    g.add_edge(15, 20, 3)
    g.add_edge(15, 19, 7)
    g.add_edge(13, 19, 2)
    g.add_edge(18, 19, 4)
    g.add_edge(18, 24, 4)
    g.add_edge(21, 24, 2)
    g.add_edge(24, 25, 7)
    g.add_edge(21, 25, 1)
    g.add_edge(20, 21, 5)
    g.add_edge(19, 20, 2)
    g.add_edge(19, 21, 5)
    g.add_edge(16, 22, 4)
    g.add_edge(21, 22, 3)
    g.add_edge(3, 4, 3)
    D1, previousVertex1 = g.dijkstra1(a)

    path = []
    cheapest_path1 = []
    key1 = b
    while True:
        if key1 == a:
            path.append(a)
            break

        else:
            path.append(key1)
            key1 = previousVertex1[key1]

    for point in path[:0:-1]:
        cheapest_path1.append(dic1[point])

    cheapest_path1 = "->".join(cheapest_path1)

    cheapest = cheapest_path1 + "->" + cheapest_path

    print(f"Distance from {dic1[int(a)]} to {dic1[int(b)]} is {D[c] + D1[b]} km ,"
          f"The optimal route is {cheapest}")


class Graph:

    def __init__(self, num_of_vertices):
        self.vertices = num_of_vertices

        self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]

        self.visited = []

    def add_edge(self, u, v, weight):

        self.edges[u][v] = weight

        self.edges[v][u] = weight

    def dijkstra(self, start_vertex):

        D = {v: float('inf') for v in range(self.vertices)}
        D[start_vertex] = 0
        pq = PriorityQueue()
        pq.put((0, start_vertex))
        previousVertex = {}
        while not pq.empty():

            (dist, current_vertex) = pq.get()

            self.visited.append(current_vertex)

            for neighbor in range(self.vertices):

                if self.edges[current_vertex][neighbor] != -1:
                    distance = self.edges[current_vertex][neighbor]

                    if neighbor not in self.visited:
                        old_cost = D[neighbor]
                        new_cost = D[current_vertex] + distance

                        if new_cost < old_cost:
                            pq.put((new_cost, neighbor))
                            D[neighbor] = new_cost
                            previousVertex[neighbor] = current_vertex

        return D, previousVertex


class Graph1:

    def __init__(self, num_of_vertices):
        self.vertices = num_of_vertices

        self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]

        self.visited = []

    def add_edge(self, u, v, weight):

        self.edges[u][v] = weight

        self.edges[v][u] = weight

    def dijkstra1(self, start_vertex):

        D1 = {v: float('inf') for v in range(self.vertices)}
        D1[start_vertex] = 0
        pq1 = PriorityQueue()
        pq1.put((0, start_vertex))
        previousVertex1 = {}
        while not pq1.empty():

            (dist, current_vertex) = pq1.get()

            self.visited.append(current_vertex)

            for neighbor in range(self.vertices):

                if self.edges[current_vertex][neighbor] != -1:
                    distance = self.edges[current_vertex][neighbor]

                    if neighbor not in self.visited:
                        old_cost = D1[neighbor]
                        new_cost = D1[current_vertex] + distance

                        if new_cost < old_cost:
                            pq1.put((new_cost, neighbor))
                            D1[neighbor] = new_cost
                            previousVertex1[neighbor] = current_vertex

        return D1, previousVertex1




def main2():
    for i in range (1,100):
        k=input("Please enter 1/-1 to enter/exit the system:")
        if k=='1':
            y=input("If you have place you must pass by,input 1,if not,print anything:")
            if y=='1':
                Dijkstra1()
            if y!='1':
                Dijkstra()
        if k=='-1':
            break


def get_token():
    #Get token
    baidu_server = "https://openapi.baidu.com/oauth/2.0/token?"
    grant_type = "client_credentials"
    client_id = "U7YIXKu0B2wsGjBu1cgaa0GF"
    client_secret = "rFdqMLA1NrTuFGiCzw9tn9WuU6uHOxUf" # YU SHUYANG's ID and keys, can change and use another one

    #get url
    url = f"{baidu_server}grant_type={grant_type}&client_id={client_id}&client_secret={client_secret}"
    res = requests.post(url)
    token = json.loads(res.text)["access_token"]
    return token


def audio_baidu(filename):
    with open(filename, "rb") as f:
        speech = base64.b64encode(f.read()).decode('utf-8')
    size = os.path.getsize(filename)
    token = get_token()
    headers = {'Content-Type': 'application/json'}
    url = "https://vop.baidu.com/server_api"
    data = {
        "format": "wav",
        "rate": "16000",
        "dev_pid": "1736",
        "speech": speech,
        "cuid": "TEDxPY",
        "len": size,
        "channel": 1,
        "token": token,
    }

    req = requests.post(url, json.dumps(data), headers)
    result = json.loads(req.text)

    if result["err_msg"] == "success.":
        print(result['result'])
        return result['result']
    else:
        print("Content acquisition failure, exit speech recognition")
        return -1


def main3():
    logging.basicConfig(level=logging.INFO)
    wav_num = 0
    while True:
        r = sr.Recognizer()
        #Enable microphone
        mic = sr.Microphone()
        logging.info('In the recording...')
        with mic as source:
            #Noise reduction
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        with open(f"00{wav_num}.wav", "wb") as f:
            #Save the recorded sound from the microphone as a wav file
            f.write(audio.get_wav_data(convert_rate=16000))
        logging.info('End of recording, identification in progress...')
        target = audio_baidu(f"00{wav_num}.wav")
        if target == -1:
            break
        wav_num += 1
        if target == ['Stop。']:
            sys.exit()


def main10():
    IMAGE_DIR_TRAINING = "C:\\Users\\25435\\PycharmProjects\\AI_final_project\\traffic_light_images\\training"
    IMAGE_DIR_TEST = "C:\\Users\\25435\\PycharmProjects\\AI_final_project\\traffic_light_images\\test"
    # need to be changed

    def load_dataset(image_dir):
        im_list = []
        image_types = ['red', 'yellow', 'green']

        for im_type in image_types:
            file_lists = glob.glob(os.path.join(image_dir, im_type, '*'))
            print(len(file_lists))
            for file in file_lists:
                im = mpimg.imread(file)

                if not im is None:
                    im_list.append((im, im_type))
        return im_list

    IMAGE_LIST = load_dataset(IMAGE_DIR_TRAINING)

    _, ax = plt.subplots(1, 3, figsize=(5, 2))
    # red
    img_red = IMAGE_LIST[0][0]
    ax[0].imshow(img_red)
    ax[0].annotate(IMAGE_LIST[0][1], xy=(2, 5), color='blue', fontsize='10')
    ax[0].axis('off')
    ax[0].set_title(img_red.shape, fontsize=10)
    # yellow
    img_yellow = IMAGE_LIST[730][0]
    ax[1].imshow(img_yellow)
    ax[1].annotate(IMAGE_LIST[730][1], xy=(2, 5), color='blue', fontsize='10')
    ax[1].axis('off')
    ax[1].set_title(img_yellow.shape, fontsize=10)
    # green
    img_green = IMAGE_LIST[800][0]
    ax[2].imshow(img_green)
    ax[2].annotate(IMAGE_LIST[800][1], xy=(2, 5), color='blue', fontsize='10')
    ax[2].axis('off')
    ax[2].set_title(img_green.shape, fontsize=10)
    plt.show()

    def standardize(image_list):
        '''
        This function takes a rgb image as input and return a standardized version
        image_list: image and label
        '''
        standard_list = []
        # Iterate through all the image-label pairs
        for item in image_list:
            image = item[0]
            label = item[1]
            # Standardize the input
            standardized_im = standardize_input(image)
            # Standardize the output(one hot)
            one_hot_label = one_hot_encode(label)
            # Append the image , and it's one hot encoded label to the full ,processed list of image data
            standard_list.append((standardized_im, one_hot_label))
        return standard_list

    def standardize_input(image):
        # Resize all images to be 32x32x3
        standard_im = cv2.resize(image, (32, 32))
        return standard_im

    def one_hot_encode(label):

        if label == 'red':
            return [1, 0, 0]
        elif label == 'yellow':
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    Standardized_Train_List = standardize(IMAGE_LIST)

    image_num = 0
    test_im = Standardized_Train_List[image_num][0]
    test_label = Standardized_Train_List[image_num][1]
    # convert to hsv
    hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)
    # Print image label
    print('Label [red, yellow, green]: ' + str(test_label))
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # Plot the original image and the three channels
    _, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].set_title('Standardized image')
    ax[0].imshow(test_im)
    ax[1].set_title('H channel')
    ax[1].imshow(h, cmap='gray')
    ax[2].set_title('S channel')
    ax[2].imshow(s, cmap='gray')
    ax[3].set_title('V channel')
    ax[3].imshow(v, cmap='gray')

    def create_feature(rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        sum_brightness = np.sum(hsv[:, :, 2])
        area = 32 * 32
        avg_brightness = sum_brightness / area  # Find the average
        return avg_brightness

    def high_saturation_pixels(rgb_image, threshold=80):
        high_sat_pixels = []
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        for i in range(32):
            for j in range(32):
                if hsv[i][j][1] > threshold:
                    high_sat_pixels.append(rgb_image[i][j])
        if not high_sat_pixels:
            return highest_sat_pixel(rgb_image)

        sum_red = 0
        sum_green = 0
        for pixel in high_sat_pixels:
            sum_red += pixel[0]
            sum_green += pixel[1]

        avg_red = sum_red / len(high_sat_pixels)
        avg_green = sum_green / len(high_sat_pixels) * 0.8
        return avg_red, avg_green

    def highest_sat_pixel(rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]

        x, y = (np.unravel_index(np.argmax(s), s.shape))
        if rgb_image[x, y, 0] > rgb_image[x, y, 1] * 0.9:
            return 1, 0  # red has a higher content
        return 0, 1

    def estimate_label(rgb_image, display=False):
        return red_green_yellow(rgb_image, display)

    def findNoneZero(rgb_image):
        rows, cols, _ = rgb_image.shape
        counter = 0
        for row in range(rows):
            for col in range(cols):
                pixels = rgb_image[row, col]
                if sum(pixels) != 0:
                    counter = counter + 1
        return counter

    def red_green_yellow(rgb_image, display):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        sum_saturation = np.sum(hsv[:, :, 1])  # Sum the brightness values
        area = 32 * 32
        avg_saturation = sum_saturation / area  # find average

        sat_low = int(avg_saturation * 1.3)  # 均值的1.3倍，工程经验
        val_low = 140
        # Green
        lower_green = np.array([70, sat_low, val_low])
        upper_green = np.array([100, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_result = cv2.bitwise_and(rgb_image, rgb_image, mask=green_mask)
        # Yellow
        lower_yellow = np.array([10, sat_low, val_low])
        upper_yellow = np.array([60, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_result = cv2.bitwise_and(rgb_image, rgb_image, mask=yellow_mask)

        # Red
        lower_red = np.array([150, sat_low, val_low])
        upper_red = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        red_result = cv2.bitwise_and(rgb_image, rgb_image, mask=red_mask)
        if display == True:
            _, ax = plt.subplots(1, 5, figsize=(20, 10))
            ax[0].set_title('rgb image')
            ax[0].imshow(rgb_image)
            ax[1].set_title('red result')
            ax[1].imshow(red_result)
            ax[2].set_title('yellow result')
            ax[2].imshow(yellow_result)
            ax[3].set_title('green result')
            ax[3].imshow(green_result)
            ax[4].set_title('hsv image')
            ax[4].imshow(hsv)
            plt.show()
        sum_green = findNoneZero(green_result)
        sum_red = findNoneZero(red_result)
        sum_yellow = findNoneZero(yellow_result)
        if sum_red >= sum_yellow and sum_red >= sum_green:
            return [1, 0, 0]  # Red
        if sum_yellow >= sum_green:
            return [0, 1, 0]  # yellow
        return [0, 0, 1]  # green

    img_test = [(img_red, 'red'), (img_yellow, 'yellow'), (img_green, 'green')]
    standardtest = standardize(img_test)

    for img in standardtest:
        predicted_label = estimate_label(img[0], display=True)
        print('Predict label :', predicted_label)
        print('True label:', img[1])

        # Using the load_dataset function in helpers.py
        # Load test data
        TEST_IMAGE_LIST = load_dataset(IMAGE_DIR_TEST)

        # Standardize the test data
        STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

        # Shuffle the standardized test data
        random.shuffle(STANDARDIZED_TEST_LIST)

    def get_misclassified_images(test_images, display=False):
        misclassified_images_labels = []
        # Iterate through all the test images
        # Classify each image  and compare to the true label
        for image in test_images:
            # Get true data
            im = image[0]
            true_label = image[1]
            assert (len(true_label) == 3), 'This true_label is not the excepted length (3).'

            # Get predicted label from your classifier
            predicted_label = estimate_label(im, display=False)
            assert (len(predicted_label) == 3), 'This predicted_label is not the excepted length (3).'

            # compare true and predicted labels
            if (predicted_label != true_label):
                # if these labels are ot equal, the image  has been misclassified
                misclassified_images_labels.append((im, predicted_label, true_label))
        # return the list of misclassified [image,predicted_label,true_label] values
        return misclassified_images_labels

    # Find all misclassified images in a given test set
    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST, display=False)
    # Accuracy calcuations
    total = len(STANDARDIZED_TEST_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total
    print('Accuracy:' + str(accuracy))
    print('Number of misclassfied images = ' + str(len(MISCLASSIFIED)) + ' out of ' + str(total))


def image_selection():
    selection = eg.buttonbox(msg='                    Please choose how to open the image', title='Face recognition',
                             choices=('Face recognition(P)', 'Camera'),images='car.gif',
                             default_choice=('Button[2]'), cancel_choice=None, callback=None, run=True)
    return selection


def select_module2():

    module1 = eg.buttonbox(title='Route planning',images='City_Map.gif',choices = 'quit')

    return module1


if __name__ == '__main__':
    #模块选择
    while(True):
        module = select_module()
        if(module == None):
            break


        elif(module == 'car3.gif'):
            pass

        elif(module == 'Image identification'):

            selection = image_selection()


            if (selection == 'Face recognition(P)'):

                paths = select_path()
                judge = face_recognition(paths)
            elif(selection == 'Camera'):
                camera_face_recognition()
            else:
                pass
            pass

        elif (module == 'Speech Recognition'):
            main3()
        elif (module == 'Route planning'):
            module1 = select_module2()
            main2()
        elif (module =='Traffic light detection'):
            main10()


        elif(module == 'None'):
            sys.exit()
        else:
            pass
