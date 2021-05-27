# OpenCV 
import cv2
# os module for working with directories
import os
# numpy for numpy arrays
import numpy as np
# pyrebase for uploading data to firebase
import pyrebase

# configuring for firebase
config = {
    "apiKey": "AIzaSyC8SG3R3H1VTVAscE94VvS870CL0pfMWoE",
    "authDomain": "casestudy-9226c.firebaseapp.com",
    "projectId": "casestudy-9226c",
    "storageBucket": "casestudy-9226c.appspot.com",
    "serviceAccount": "serviceAccountKey.json",
    "databaseURL": ""
}

# initializing firebase_storage object
firebase_storage = pyrebase.initialize_app(config)

# creating only storage object because we are using only fire storage
storage = firebase_storage.storage()

# subjects dictionary to store all the student class data
subjects = dict()

# student class 
class Student:
    def __init__(self,rollno,name,year,branch,section):
        self.rollno = rollno
        self.name = name
        self.year = year
        self.branch = branch
        self.section = section

# dectecting face from an image
def detect_face(img):
    # converting BGR images to GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # creating a CascadeClassifier Object which is used for face detection
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    # if no faces found
    if (len(faces) == 0):
        return None, None
    # considering only one face
    (x, y, w, h) = faces[0]
    
    # returning the face region on the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    
    # listing all folders in the traiing data folder
    dirs = os.listdir(data_folder_path)
    # faces list for appending faces
    faces = []
    # labels list for appending labels
    labels = []
    # for each folder in training folder
    for dir_name in dirs:
        # storing label as dir_name which is roll no of student
        label = int(dir_name)
        # path for the roll number folder of the student
        subject_dir_path = data_folder_path + "/" + dir_name
        # listing all the images in roll number folder
        subject_images_names = os.listdir(subject_dir_path)
        
        # for each image
        for image_name in subject_images_names:
            # checking if any system files are present
            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name
            # reading the image
            image = cv2.imread(image_path)
            
            # displaying 
            """
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            """
            
            # detecing faces 
            face, rect = detect_face(image)

            # if face found
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    # faces list and labels list
    return faces, labels

# function to collect training data for each student
def collect_images(student):

    # list to check whether roll number already exists
    rolllist = []

    # reading subjects file and appending to rolllist
    f = open("subjects.txt",'r')
    while(1):
        l = f.readline()
        if not l:
            break
        else:
            temp = l.split(';')
            rolllist.append(temp[0])
    f.close()

    # if roll number exists then return 0
    if student.rollno in rolllist:
        return 0
    
    # writing new student data to subject file
    f = open("subjects.txt",'a')
    f.write(student.rollno + ';' + student.name + ';' + student.year + ';' + student.branch + ';' + student.section + '\n')
    f.close()

    # creating to folder(rollno) in training-data 
    path = os.path.join("training-data",student.rollno)
    os.mkdir(path)

    # capturing from webcam
    cam = cv2.VideoCapture(0)
    img_counter = 1

    # capturing 12 images
    while(img_counter<13):
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test",frame)
        k = cv2.waitKey(1)

        # for escape keystroke 
        if k%256 == 27:
            break
        
        # for space keystroke
        elif k%256 == 32:
            img_name = "{}.jpg".format(img_counter)
            cv2.imwrite("%s/%s"%(path,img_name),frame)
            img_counter += 1

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return 1
    
# function to collect the test data using webcam for each event
def addtestimage(event,subjects,face_recognizer):

    # creating event folder in event path
    os.mkdir("test-data/"+event)
    cam = cv2.VideoCapture(0)
    img_counter = 0
    while(1):
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test",frame)
        k = cv2.waitKey(1)

        if k%256 == 27:
            break
        
        elif k%256 == 32:
            img_name = "test{}.jpg".format(img_counter)
            cv2.imwrite("%s/%s"%("test-data/" + event + "/",img_name),frame)
            img_counter += 1
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    face_recognizer.train(faces, np.array(labels))

    print("Predicting images...")

    eventpath = os.path.join("test-data",event)

    imgs = os.listdir(eventpath)

    imgcount = len(imgs)

    attendance_list = dict()

    for i in range(imgcount):
        path = "test-data/" + event + "/test" + str(i) + ".jpg"
        test_img = cv2.imread(path)
        predict_img, label, confidence = predict(test_img,subjects)
        if label in subjects:
            cv2.imshow(subjects[label][0], cv2.resize(predict_img, (400, 500)))
            key = subjects[label][0]
            key = int(key)
            attendance_list[key] = subjects[label]
            print("accuracy:",100 - confidence)
    

    datafield = ['rollno: ','name: ','year: ','branch: ','section: ']
    f = open("attendance/"+event+'.txt',"a")
    for data in attendance_list:
        f.write(datafield[0] + attendance_list[data][0] + ' ' + datafield[1] + attendance_list[data][1] + ' ' + datafield[2] + attendance_list[data][2] + ' ' + datafield[3] + attendance_list[data][3] + ' ' + datafield[4] + attendance_list[data][4] + '\n')
    f.close()

    event_path = os.path.join("attendance",event+str(".txt"))

    storage.child(event).put(event_path)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
        
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

def predict(test_img,subjects):
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label][0]
    
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img,label,confidence

# collecting student details and collecting the training data
def n1():
    roll = input("Enter Roll no: ")
    name = input("Enter name: ")
    year = input("Enter year: ")
    branch = input("Enter Branch: ")
    section = input("Enter section: ")
    student = Student(roll,name,year,branch,section)
    # calling function which collects the training data.
    flag = collect_images(student)
    if flag == 0:
        print("Rollno already Exists. Try different rollno ")
        n1()

# collecting test data for each event
def n2(face_recognizer):
    events_list = os.listdir("test-data")
    event = input("Enter Event name: ")
    if event in events_list:
        print("Event Already Exists")
        n2(face_recognizer)
    subjects = dict()
    f = open("subjects.txt",'r')
    while(1):
        x = f.readline()
        if x == "":
            break
        x = x.split(';')
        subjects[int(x[0])] = [x[0],x[1],x[2],x[3],x[4][:-1]]
    f.close()
    # calling function which collects the test data for each event.
    addtestimage(event,subjects,face_recognizer)

# collecting data from user
n = int(input("Enter 1 if you want train or Enter 2 if you want to test: "))
if n == 1:
    n1()
if n == 2:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    n2(face_recognizer)


   