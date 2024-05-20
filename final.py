import cv2
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import soundfile
import speech_recognition as sr
import pyttsx3


net = cv2.dnn.readNet("H:/CE Subjects/597_Project/Open-CV-main/DetectByAudio/yolov4-tiny.cfg", 
                      "H:/CE Subjects/597_Project/Open-CV-main/DetectByAudio/yolov4-tiny.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

classNames = []
df = pd.read_csv("H:/CE Subjects/597_Project/Open-CV-main/DetectByAudio/classes.txt", header=None, names=["ClassName"])
for index, row in df.iterrows():
    ClassName = df.iloc[index]['ClassName']
    classNames.append(ClassName)

engine = pyttsx3.init()

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

x1 = 20 
y1 = 20 
x2 = 570 
y2 = 90 

fs = 44100  
seconds = 3  
audio_file_name = "c:/temp/output.wav"

ButtonFlag = False
LookForThisClassName = ""


def recognize_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print(f"Voice command recognized: {command}")
        return command.lower()  
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print(f"Error fetching results from Google Speech Recognition service: {e}")
    return ""


def getTextFromAudio():
   
    data, samplerate = soundfile.read(audio_file_name)
    soundfile.write('c:/temp/outputNew.wav', data, samplerate, subtype='PCM_16')

    
    recognizer = sr.Recognizer()
    jackhammer = sr.AudioFile('c:/temp/outputNew.wav')

    with jackhammer as source: 
        audio = recognizer.record(source)

    result = recognizer.recognize_google(audio)

    print(result)
    return result 

def recordAudioByMouseClick(event, x, y, flags, params):
    global ButtonFlag 
    global LookForThisClassName

    if event == cv2.EVENT_LBUTTONDOWN:
        if x1 <= x <= x2 and y1 <= y <= y2: 
            print("Click inside")

           
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
            sd.wait()  
            write(audio_file_name, fs, myrecording)  

            LookForThisClassName = getTextFromAudio()

            if ButtonFlag is False: 
                ButtonFlag = True
        else: 
            print("Clicked outside")
            ButtonFlag = False

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", recordAudioByMouseClick)
    

while True:
   
    ret, frame = cap.read()

    
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    (class_ids, scores, bbox) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids, scores, bbox):
        
        x, y, width, height = bbox 
        name = classNames[class_id]

        index = LookForThisClassName.find(name) 

        if ButtonFlag is True and index > 0: 
            cv2.rectangle(frame, (x, y), (x + width, y + height), (130, 50, 50), 3)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (120, 50, 50), 2)
           
            engine.say(f"Detected {name}")
            engine.runAndWait()

    cv2.rectangle(frame, (x1, y1), (x2, y2), (153, 0, 0), -1)
    cv2.putText(frame, "Click for record -3 seconds", (40, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cv2.destroyAllWindows()
cap.release()
