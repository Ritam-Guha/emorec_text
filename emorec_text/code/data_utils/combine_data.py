import emorec_text.config as config
from emorec_text.code.utils.path_utils import create_dir

import pandas as pd
import os
import copy


def combine_data():
    create_dir("data/text_emotion")
    list_video_emotions = os.listdir(f"{config.BASE_PATH}/data/video_emotions")
    list_transcriptions = os.listdir(f"{config.BASE_PATH}/data/transcription_storage")
    list_common_videos = list(set(list_transcriptions).intersection(set(list_video_emotions)))

    for video_id in list_common_videos:
        try:
            df_video_emotions = pd.read_csv(f"{config.BASE_PATH}/data/video_emotions/{video_id}")
            df_transcription = pd.read_csv(f"{config.BASE_PATH}/data/transcription_storage/{video_id}")

            # get the emotion
            df_video_emotions = df_video_emotions[["frame", "dominant_emotion"]]
            df_video_emotions["frame"] = df_video_emotions["frame"].str.extract('(\d+)').astype(int)
            df_video_emotions = df_video_emotions.sort_values(by="frame")

            # majority voting in emotions
            df_emotion = copy.deepcopy(df_transcription)
            df_emotion["end"] = df_emotion["start"] + df_emotion["duration"]

            start_time = list(df_emotion["start"])
            end_time = list(df_emotion["end"])
            list_dominant_emotions = []

            for (start, end) in zip(start_time, end_time):
                cur_start_time = int(start)
                cur_end_time = int(end)
                list_cur_dominant_emotions = list(df_video_emotions[df_video_emotions["frame"].between(cur_start_time, cur_end_time)]["dominant_emotion"])

                try:
                    list_dominant_emotions.append(max(set(list_cur_dominant_emotions), key = list_dominant_emotions.count))
                except:
                    list_dominant_emotions.append("NA")

            df_emotion["dominant_emotion"] = list_dominant_emotions
            df_emotion.to_csv(f"{config.BASE_PATH}/data/text_emotion/{video_id}", index=False)
            print(f"{video_id} done...")

        except:
            continue


def main():
    combine_data()


if __name__ == "__main__":
    main()

