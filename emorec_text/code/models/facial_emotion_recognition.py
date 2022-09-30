import emorec_text.config as config

from deepface import DeepFace
from fer import FER
import matplotlib.pyplot as plt
import os
import cv2


class FaceEmoRec:
    def __init__(self,
                 model_type="deepface"):
        self.model_type = model_type

    @staticmethod
    def get_emotion(image):
        # execute the model of choice
        # if self.model_type == "deepface":
        #     face_analysis = DeepFace.analyze(img_path=image_path)
        #     dominant_emotion = face_analysis["dominant_emotion"]
        #
        # elif self.model_type == "fer":
        # test_image = plt.imread(image_path)
        emo_detector = FER(mtcnn=True)
        captured_emotions = emo_detector.detect_emotions(image)
        dominant_emotion, emotion_score = emo_detector.top_emotion(image)
        return dominant_emotion

    def apply_video(self,
                    video_path):
        video = cv2.VideoCapture(video_path);

        while True:
            check, frame = video.read()
            frame, faces = self.capture_face(frame)
            dominant_emotion = self.get_emotion(frame)
            for x, y, w, h in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3);

            # add emotion
            cv2.putText(img=frame, text=dominant_emotion, org=(x + int(w / 10), y + int(h / 1.5)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=1)
            cv2.imshow('Face Detector', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    def apply_image(self,
                    image_path):

        frame = cv2.imread(image_path)
        dominant_emotion = self.get_emotion(frame)

        # draw the bounding boxes
        frame, faces = self.capture_face(image=frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # add emotion
        cv2.putText(img=frame, text=dominant_emotion, org=(x + int(w / 10), y + int(h / 1.5)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=1)
        cv2.imshow("frame", frame)
        cv2.waitKey(0)

    @staticmethod
    def capture_face(image):
        faceCascade = cv2.CascadeClassifier(f'{config.BASE_PATH}/data/haarcascade_frontalface_default.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return image, faces


def main():
    model_type = "fer"
    face_emo_rec = FaceEmoRec(model_type=model_type)
    test_image_path = f"{config.BASE_PATH}/data/frame_storage/2_-GiiulAw8/frame_11.jpg"
    test_video_path = f"{config.BASE_PATH}/data/video_storage/2_-GiiulAw8.mp4"
    # face_emo_rec.apply_image(image_path=test_image_path)
    face_emo_rec.apply_video(video_path=test_video_path)
    # face_emo_rec.capture_face(image_path=test_image_path)


if __name__ == "__main__":
    main()