import emorec_text.config as config
from emorec_text.code.utils.path_utils import create_dir

import os
import cv2


class VideoToImage:
    def __init__(self,
                 video_path):
        self.video_path = video_path
        self.list_videos = os.listdir(self.video_path)

    def convert_to_image(self):
        # create the directory
        create_dir("data/frames", delete=True)
        for video_name in self.list_videos:
            # create the place for storing frames
            create_dir(f"data/frames/{video_name.split('.', 1)[0]}")

            video_cap = cv2.VideoCapture(f"{self.video_path}/{video_name}")
            success, image = video_cap.read()
            count = 0
            while success:
                cv2.imwrite(f"{config.BASE_PATH}/data/frames/{video_name.split('.', 1)[0]}/frame_{count}.jpg", image)  # save frame as JPEG file
                success, image = video_cap.read()
                count += 1

            if success:
                print(f"successfully extracted frames from {video_name}")


def main():
    video_path = f"{config.BASE_PATH}/data/videos"
    v2i = VideoToImage(video_path=video_path)

    v2i.convert_to_image()


if __name__ == "__main__":
    main()