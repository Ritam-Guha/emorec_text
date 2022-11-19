import emorec_text.config as config
from emorec_text.code.utils.path_utils import create_dir

import pandas as pd
import os
import copy
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def combine_transcripts(df_transcription,
                        n_comb=5):
    df_emotion = pd.DataFrame(columns=["text", "start", "duration"])
    num_rows = int(np.ceil(df_transcription.shape[0]/n_comb))
    for i in range(num_rows):
        start = df_transcription.iloc[i*n_comb]["start"]
        end_idx = (i+1)*n_comb-1 if (i+1)*n_comb < df_emotion.shape[0] else df_emotion.shape[0] - 1
        end = df_transcription.iloc[end_idx]["start"] + df_transcription.iloc[end_idx]["duration"]
        duration = end-start
        text = " ".join(list(df_transcription.iloc[i*n_comb:(i+1)*n_comb]["text"]))
        df_emotion = df_emotion.append({"start": start, "duration": duration, "text": text}, ignore_index=True)

    return df_emotion


def combine_data():
    create_dir("data/text_emotion")
    list_video_emotions = os.listdir(f"{config.BASE_PATH}/data/video_emotions")
    list_transcriptions = os.listdir(f"{config.BASE_PATH}/data/transcription_storage")
    list_common_videos = list(set(list_transcriptions).intersection(set(list_video_emotions)))

    for video_id in list_common_videos:
        # read the emotion files and transcriptions
        df_video_emotions = pd.read_csv(f"{config.BASE_PATH}/data/video_emotions/{video_id}")
        df_transcription = pd.read_csv(f"{config.BASE_PATH}/data/transcription_storage/{video_id}")
        df_video_emotions.fillna("NA", inplace=True)

        # get the emotion
        df_video_emotions = df_video_emotions[["frame", "dominant_emotion"]]
        df_video_emotions["frame"] = df_video_emotions["frame"].str.extract('(\d+)').astype(int)
        df_video_emotions = df_video_emotions.sort_values(by="frame")

        # majority voting in emotions
        # df_emotion = combine_transcripts(df_transcription)
        df_emotion = copy.deepcopy(df_transcription)
        df_emotion["end"] = df_emotion["start"] + df_emotion["duration"]

        start_time = list(df_emotion["start"])
        end_time = list(df_emotion["end"])
        list_dominant_emotions = []

        for (start, end) in zip(start_time, end_time):
            # collect the dominant emotion from start_time to end_time
            cur_start_time = int(start)
            cur_end_time = int(end)
            list_cur_dominant_emotions = list(df_video_emotions[df_video_emotions["frame"].between(cur_start_time,
                                                                                                   cur_end_time)]["dominant_emotion"])

            if len(set(list_cur_dominant_emotions)) > 0:
                list_dominant_emotions.append(max(set(list_cur_dominant_emotions), key=list_dominant_emotions.count))
            else:
                list_dominant_emotions.append("NA")

        df_emotion["dominant_emotion"] = list_dominant_emotions
        df_emotion.to_csv(f"{config.BASE_PATH}/data/text_emotion/{video_id}", index=False)
        print(f"{video_id} done...")


def main():
    combine_data()


if __name__ == "__main__":
    main()

