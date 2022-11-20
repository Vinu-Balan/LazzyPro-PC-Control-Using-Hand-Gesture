import numpy as np
import pickle
import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)

model_file_name = 'lips_model.sav'
model = pickle.load(open(model_file_name, 'rb'))

mp_face_mesh = mp.solutions.face_mesh

face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                         min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles

font = cv2.FONT_HERSHEY_SIMPLEX
org = (185, 50)
color = (0, 0, 255)

sTime = time.time()
cur = time.time()

row_list = []
pivot_x = 0
pivot_y = 0

while (cap.isOpened()):
    ret, img = cap.read()
    img = cv2.resize(img, (960, 540))
    x, y, c = img.shape

    t_elapsed = abs(sTime - time.time())

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh_images.process(image=imgRGB)
    if results.multi_face_landmarks != 0:
        if results.multi_face_landmarks:
            for handlms in results.multi_face_landmarks:
                for id, lms in enumerate(handlms.landmark):
                    # print(id,lms)
                    h, w, c = img.shape
                    # print(img.shape)
                    cx, cy = int(lms.x * w), int(lms.y * h)
                    # print("id:",id,", x:",cx,", y:",cy)
                    if id == 49:
                        pivot_x = int(lms.x * x)
                        pivot_y = int(lms.y * y)
                        # mp_drawing.draw_landmarks(img, handlms, mp_face_mesh.FACEMESH_TESSELATION)
                        row_list.append(str(pivot_x))
                        row_list.append(str(pivot_y))
                    elif id > 49 and id <= 68:
                        lmx = int(lms.x * x)
                        lmy = int(lms.y * y)
                        # mp_drawing.draw_landmarks(img, handlms, mp_face_mesh.FACEMESH_TESSELATION)

                        row_list.append(str(pivot_x - lmx))
                        row_list.append(str(pivot_y - lmy))
                break
            # Predict gesture
            prediction = model.predict([row_list])
            # print(prediction)
            classID = np.argmax(prediction)
            #print(prediction[0])
            cv2.putText(img, str(prediction[0]), org, font, fontScale=1, thickness=2, color=color)
            row_list = []
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # if the 'q' is pressed quit.'OxFF' is for 64 bit.[if waitKey==True] is condition
                break
            if ret == True:
                cv2.imshow("result", img)

cap.release()
cv2.destroyAllWindows()