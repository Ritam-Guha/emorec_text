import emorec_text.config as config

from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
from urllib.parse import urlparse, parse_qs
import argparse

parser = argparse.ArgumentParser("transcription")
parser.add_argument("--url",
                    default="https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z",
                    type=str, help="url of the youtube video")


class YouTubeTranscriptor:
    def __int__(self):
        """
        class to store youtube transcriptions
        """
        self.type_downloader = "transcript"

    def store_transcript(self,
                         url,
                         languages=["en"]):
        """
        :param url: url for the youtube video
        :param languages: languages for which the transcriptions should be stored
        """
        video_id = self.get_video_id(url)
        transcription = YouTubeTranscriptApi.get_transcript(video_id=video_id,
                                                  languages=languages)

        df = pd.DataFrame.from_dict(transcription)
        df.to_csv(f"{config.BASE_PATH}/transcription_storage/{video_id}.csv", sep=",", index=False)

    @staticmethod
    def get_video_id(url):
        """
        Returns Video_ID extracting from the given url of Youtube
        """

        if url.startswith(('youtu', 'www')):
            url = 'http://' + url

        query = urlparse(url)

        if 'youtube' in query.hostname:
            if query.path == '/watch':
                return parse_qs(query.query)['v'][0]
            elif query.path.startswith(('/embed/', '/v/')):
                return query.path.split('/')[2]
        elif 'youtu.be' in query.hostname:
            return query.path[1:]
        else:
            raise ValueError


def main():
    args = parser.parse_args()
    url = args.url
    transcriptor = YouTubeTranscriptor()
    transcriptor.store_transcript(url=url)


if __name__ == "__main__":
    main()