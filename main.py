import cv2
import numpy as np
from datetime import datetime
import sqlite3
import time

files = ['video/file1.mp4', 'video/file2.mp4', 'video/file3.mp4', 'video/file4.mp4', 'video/file5.mp4', 'video/file6.mp4', 'video/file7.mp4', 'video/file8.mp4', 'video/file9.mp4', ]

people_amounts = []  # a list of number of people, its average is a number of people in the room,
                     # refreshes every <settings.TIMER> minutes


#work with db
def get_settings():
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM Settings")
    a = dict(cursor.fetchall())
    connection.commit()
    connection.close()
    return a

def write_data(time, num, mode, cam):
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()
    cursor.execute("INSERT INTO Data (measure_time, avg_people, mode, camera) VALUES (?, ?, ?, ?)", (time, num, mode, cam))
    connection.commit()
    connection.close()


#Work with people data

def average():
    global people_amounts
    return sum(people_amounts) / len(people_amounts) if people_amounts != [] else 0

def refresh(delta_time):
    global people_amounts
    if delta_time >= settings['timer']:
        if settings['save_data']:
            write_data()
        

        people_amounts = []


#Work with image
def start_video(file):
    """
    file: 0 - camera
          'path/to/video.mp4' - video file
    """
    cap = cv2.VideoCapture(file)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15.,
(settings['resx'], settings['resy'])) if settings['save_video'] else None
    return cap, out


def draw_data(frame, camera, num_people):
    cv2.rectangle(frame, (0, settings['resy']-20), (settings['resx'], settings['resy']), (255, 0, 0), -1)
    string = f'camera: {camera}, number of people: {num_people}. Print Esc to exit program, W or Q to switch cameras'
    cv2.putText(frame,  
            string,  
            (5, settings['resy']-5),  
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,  
            (255, 255, 255),  
            2) 

def draw_weight(frame, x, y, weight):
    cv2.rectangle(frame, (x, y+10), (x+30, y), (255, 0, 0), -1)
    cv2.putText(frame,  
                str(round(weight, 2)),  
                (x+2, y+7),  
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,  
                (255, 255, 255),  
                1) 


def draw_boxes(frame, boxes, weights):
    for i, (xA, yA, xB, yB) in enumerate(boxes):
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (255, 0, 0), 2)
        draw_weight(frame, xA, yA, weights[i])
        


def display_image(frame):
    cv2.imshow('press Esc to exit', frame)

def close_image(cap, out):
    cap.release
    if settings['save_video']:
        out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)





settings = get_settings()

if settings['print_data']:
    print(settings)



# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if settings['save_video']: cv2.startWindowThread()

j = 0
while True:
    j %= len(files)
    file = files[j]
    cap, out = start_video(file)

    previous_time = time.time()

    while cap.isOpened():
        current_time = time.time()

        if current_time - previous_time >= settings['timer']:
            if settings['save_data']:
                current_datetime = datetime.now()
                write_data(current_datetime.strftime("%Y-%m-%d %H:%M:%S"), average(), 0, file)
                if settings['print_data']:
                    print('avg: ', end='')
                    print(average())
                    print("DATA SENT")
            previous_time = time.time()
            people_amounts = []
        
        ret, frame = cap.read()
        if not ret:
            break
        # prepare image for better recognition
        frame = cv2.resize(frame, (settings['resx'], settings['resy']))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))
        #$test

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        people_amounts.append(len(boxes))

        draw_data(frame, file, len(boxes))
        draw_boxes(frame, boxes, weights)
        if settings['save_video']:
            out.write(frame.astype('uint8'))

        if settings['display_video']:
            display_image(frame)

        if settings['print_data']:
            print(len(boxes))
            print(weights)

        w = ord('w')
        q = ord('q')
        key = cv2.waitKey(1)
        if key == q:
            j -= 1
            break
        if key == w:
            j += 1
            break
        if key == 27:
            close_image(cap, out)
            exit()
           



close_image(cap, out)