import cv2
import mediapipe as mp
import time
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

device = 0 # camera device number
bimg = cv2.imread("./imgs/space.png")
bimg = cv2.resize(bimg, dsize=(640,480))

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now
#point = [top,center,under,left,right]
def combi(bimg, img, point):
    white = np.ones((bimg.shape), dtype=np.uint8) * 255
    min_x = point[3][0]
    max_x = point[4][0]
    if min_x>point[1][0]:
        min_x=point[1][0]
    if max_x<point[1][0]:
        max_x=point[1][0]

    white[point[0][1]:point[2][1],min_x:max_x] = bimg[point[0][1]:point[2][1],min_x:max_x]
    dwhite = white
    img[dwhite==[255, 255, 255]] = bimg[dwhite==[255, 255, 255]]

def drawFace(img, landmarks):
    image_width, image_height = img.shape[1], img.shape[0]
    landmark_point = []
    global bimg
    
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])
  
    if len(landmark_point) != 0:
        point = [landmark_point[10],landmark_point[19],landmark_point[152],landmark_point[234],landmark_point[454]]
        combi(bimg, img, point)
        bimg = bimg
    """
        #top
        cv2.circle(img,(landmark_point[10][0],landmark_point[10][1] ), 7, (255,0,0), 3)
        #left
        cv2.circle(img,(landmark_point[234][0],landmark_point[234][1] ), 7, (255,255,0), 3)
        #right
        cv2.circle(img,(landmark_point[454][0],landmark_point[454][1] ), 7, (0,0,0), 3)
        #center
        cv2.circle(img,(landmark_point[19][0],landmark_point[19][1] ), 7, (0,255,0), 3)
        #under
        cv2.circle(img,(landmark_point[152][0],landmark_point[152][1] ), 7, (0,0,255), 3)
    """
       
    

def main():
    
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            frame_now=getFrameNumber(start, fps)
            if frame_now == frame_prv:
                continue
            frame_prv = frame_now

            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = face_mesh.process(frame)

            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    drawFace(frame, face_landmarks)
            
            cv2.imshow('virtual background', frame)
            #cv2.imshow("background",bimg)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

if __name__ == '__main__':
    main()
