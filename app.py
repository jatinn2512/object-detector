import argparse
import time
import cv2
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=0, help='0 for webcam or path to video/file')
    p.add_argument('--imgsz', type=int, default=320)
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--frame-skip', type=int, default=1)
    p.add_argument('--save', action='store_true')
    p.add_argument('--classes', nargs='+', type=int, default=[0])
    return p.parse_args()

class SimpleDetector:
    def __init__(self, model_name='yolov8n.pt', device='cpu', img_size=320, conf=0.35, classes=[0]):
        self.model = YOLO(model_name)
        self.device = device
        self.img_size = img_size
        self.conf = conf
        self.classes = classes

    def run(self, source=0, frame_skip=1, save=False):
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        writer = None
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter('detections_output.mp4', fourcc, 10, (640, 480))

        frame_count, last_time = 0, time.time()

        try:
            stream = self.model.predict(
                source=source, stream=True, device=self.device,
                conf=self.conf, imgsz=self.img_size, classes=self.classes
            )
            for result in stream:
                if frame_count % (frame_skip + 1) != 0:
                    frame_count += 1
                    continue

                annotated = result.plot()
                if writer is not None:
                    writer.write(annotated)
                cv2.imshow('YOLOv8 CPU', annotated)

                now = time.time()
                fps = 1.0 / (now - last_time) if now != last_time else 0.0
                last_time = now
                cv2.putText(annotated, f'FPS:{fps:.1f}', (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                frame_count += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    detector = SimpleDetector(
        model_name='yolov8n.pt',
        device='cpu',
        img_size=args.imgsz,
        conf=args.conf,
        classes=args.classes
    )
    detector.run(source=args.source, frame_skip=args.frame_skip, save=args.save)
