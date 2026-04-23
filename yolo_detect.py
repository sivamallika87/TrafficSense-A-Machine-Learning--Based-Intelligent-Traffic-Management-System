import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# For Streamlit app
def get_vehicle_count():
    cap = cv2.VideoCapture("traffic.mp4")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return 0

    results = model(frame)

    count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            if cls in [2, 3, 5, 7]:
                count += 1

    return count


# For direct run (video display)
if __name__ == "__main__":

    cap = cv2.VideoCapture("traffic.mp4")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls in [2,3,5,7]:
                    count += 1

        frame = results[0].plot()

        cv2.putText(frame, f"Vehicle Count: {count}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("TrafficSense YOLO", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()