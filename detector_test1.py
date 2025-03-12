import logging as log 
import time
from time import perf_counter
import cv2
from openvino.runtime import Core, get_version 
from landmarks_detector import LandmarksDetector 
from face_detector import FaceDetector 
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from model_api.performance_metrics import PerformanceMetrics
import psycopg2
from psycopg2 import Error
from datetime import datetime, timedelta

DESIRED_FPS = 24


source = "videos/M2 tester.mp4"
device = 'CPU'
faceDETECT = "model_2022_3/face-detection-retail-0005.xml"
faceLANDMARK = "model_2022_3/landmarks-regression-retail-0009.xml"
faceIDENTIFY = "model_2022_3/face-reidentification-retail-0095.xml"

# Define a global counter to generate unique table names
table_counter = 1
last_table_creation_time = datetime.now()

# # Establish a connection to the PostgreSQL database
# try:
#     conn = psycopg2.connect(
#         host='localhost',
#         database='attendance',
#         user='postgres',
#         password='Sanskar@12'
#     )
# except Error as e:
#     print(f"Error connecting to PostgreSQL: {e}")
#     conn = None

# Function to create a new table for attendance records with a unique name
def create_new_attendance_table(connection):
    global table_counter, last_table_creation_time
    if connection is not None:
        cursor = connection.cursor()
        try:
            table_name = f"attendance_{table_counter}"
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (name VARCHAR, time TIME, date DATE)")
            connection.commit()
            print(f"Created table: {table_name}")
            last_table_creation_time = datetime.now()
            table_counter += 1
        except Exception as e:
            print(f"Error creating table: {e}")
            connection.rollback()
        cursor.close()

# Function to mark attendance in the specified table
def markAttendanceInTable(name, connection, table_name):
    if connection is not None:
        cursor = connection.cursor()

        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')

        try:
            cursor.execute("SELECT COUNT(*) FROM {} WHERE name = %s AND date = %s".format(table_name), (name, date))
            attendance_count = cursor.fetchone()[0]

            if attendance_count == 0:
                cursor.execute(f"INSERT INTO {table_name} (name, time, date) VALUES (%s, %s, %s)", (name, time, date))
                connection.commit()
                print(f"Attendance recorded for {name} at {time} on {date} in table {table_name}")
            # else:
                # print(f"{name} has already been marked as present today.")
        except Exception as e:
            print(f"Error inserting attendance record: {e}")
            connection.rollback()

        cursor.close()

class FrameProcessor:
    QUEUE_SIZE = 16
    def __init__(self,):
        log.info('openVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()
        self.face_detector = FaceDetector(core, faceDETECT, input_size = (0, 0), confidence_threshold=0.8)
        self.landmarks_detector = LandmarksDetector(core, faceLANDMARK)
        self.face_identifier = FaceIdentifier(core, faceIDENTIFY, match_threshold = 0.7, match_algo = 'HUNGARIAN')
        self.face_detector.deploy(device)
        self.landmarks_detector.deploy(device, self.QUEUE_SIZE)
        self.face_identifier.deploy(device, self.QUEUE_SIZE)
        self.faces_database = FacesDatabase('face_img', self.face_identifier, self.landmarks_detector,)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

    def face_process(self, frame):
        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE > len(rois):
            rois = rois[:self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        return [rois, landmarks, face_identities]

def draw_face_detection(frame, frame_processor, detections):
    size = frame.shape[:2]
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
        face_point = xmin, ymin 
        for point in landmarks:
            x = int(xmin + roi.size[0] * point[0])
            y = int(ymin + roi.size[1] * point[1])
            cv2.circle(frame, (x,y), 1, (0, 255, 255), 2)
        image_recognizer(frame, text, identity, face_point, 0.75)
    return frame

def image_recognizer(frame, text, identity, face_point, threshold):
    xmin, ymin = face_point
    if identity.id != FaceIdentifier.UNKNOWN_ID:
        if (1-identity.distance) > threshold:
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,0), 1)
            # Here you need to mark attendance using the name variable, adjust it accordingly
            # markAttendanceInTable(text, conn, table_name)
        else:
            textsize = cv2.getTextSize("unknown", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, "unknown", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0 ,0), 1)

cap = cv2.VideoCapture(source)
frame_delay = int(1000 / DESIRED_FPS)

frame_processor = FrameProcessor()
metrics = PerformanceMetrics()

# Initialize table_name before the loop
table_name = f"attendance_{table_counter - 1}"


while True:
    start_time = perf_counter()
    ret, frame = cap.read()
    
    time1 = time.time()
    detections = frame_processor.face_process(frame)
    frame = draw_face_detection(frame, frame_processor, detections)
    metrics.update(start_time, frame)

    # Check if it's time to create a new attendance table
    if (datetime.now() - last_table_creation_time) >= timedelta(minutes=5):
        print("Calling create_new_attendance_table")
        # create_new_attendance_table(conn)
        table_name = f"attendance_{table_counter - 1}"  # Update table_name after creating a new table

    time2 = time.time()
    timediff = time2 - time1

    sleeptime = max(0, (frame_delay / 1200) - timediff)
    time.sleep(sleeptime)
    
    cv2.imshow("face recognition demo", frame)
    key = cv2.waitKey(1)
    if key in {ord('q'), ord('Q'), 27}:
        cap.release()
        cv2.destroyAllWindows()
        break
