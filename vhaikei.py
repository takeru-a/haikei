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

def landmarkchecker(p1,p2, landmarks):
    get_point = []
    flag = False
    for point in landmarks:
        if p1<=point[1]<=p2:
            get_point.append(point[0])
    if len(get_point)>=2:
        flag = True
    return flag

def get_landmark(p1,p2, landmarks):
    get_point = []
    for point in landmarks:
        if p1<=point[1]<=p2:
            get_point.append(point[0])
    
    get_point.sort()
    result = [get_point[0],get_point[len(get_point)-1]]
    return result
    


#point = [top,center,under,left,right]
def combi(bimg, img, point):
    white = np.ones((bimg.shape), dtype=np.uint8) * 255
    top = point[10][1]
    under = point[152][1]
    start_y = top
    end_y = start_y + 1
    min_x = point[3][0]
    max_x = point[4][0]
    for i in range(top, under+1):
        if landmarkchecker(start_y, end_y, point):
            x_points = get_landmark(start_y, end_y, point)
            min_x, max_x = x_points[0],x_points[1]
            white[start_y:end_y,min_x:max_x] = bimg[start_y:end_y,min_x:max_x]
            dwhite = white
            #28回に１回スタート位置が変更する
            if end_y%28==0:
                start_y = end_y
        end_y += 1
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
        """
        for i in range(0, len(landmark_point)):
            cv2.circle(img, (int(landmark_point[i][0]),int(landmark_point[i][1])), 1, (0, 255, 0), 1)
            """
        combi(bimg, img, landmark_point)
     
       
    

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
