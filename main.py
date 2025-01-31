import cv2

face_ref = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
  optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  return faces

def draw_rectangle(frame):
  for (x, y, w, h) in face_detection(frame):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
    
def close_windows():
  camera.release()
  cv2.destroyAllWindows()
  exit()
    
def main():
  while True:
    _, frame = camera.read()
    draw_rectangle(frame)
    cv2.imshow("Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      close_windows()
    
if __name__ == "__main__":
  main()

# while True:
#   _, frame = camera.read()
#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   faces = face_ref.detectMultiScale(gray, 1.3, 5)
#   for (x, y, w, h) in faces:
#     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = frame[y:y+h, x:x+w]
#   cv2.imshow('frame', frame)
  
  