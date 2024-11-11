import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("C:\\Users\\Surya S\\Downloads\\yolov3.weights", "C:\\Users\\Surya S\\Downloads\\yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (COCO dataset)
with open("C:\\Users\\Surya S\\Downloads\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the video capture (or load an image)
cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a file path

while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the image for the YOLO model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Post-processing: draw bounding boxes for detected objects
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                # Get the bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to avoid overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = [0, 255, 0]  # Green color for bounding box

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output
    cv2.imshow("Object Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
