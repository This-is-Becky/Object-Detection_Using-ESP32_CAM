import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("../yolo_weights/yolov8n.pt")

url = 'http://......./cam-lo.jpg'

while True:
    # Fetch the image from the ESP32-CAM
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    # Perform object detection
    results = model(im)

    # Draw bounding boxes and labels
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates of the box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label_id = int(box.cls[0])
            confidence = box.conf[0]
            label = model.names[label_id]

            # Draw the bounding box
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('ESP32-CAM Feed', im)

    # Exit if the window is closed
    if cv2.getWindowProperty('ESP32-CAM Feed', cv2.WND_PROP_VISIBLE) < 1:
        break

    cv2.waitKey(1)

# Close all windows
cv2.destroyAllWindows()
