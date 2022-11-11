# Online Multi-modal Engagement Analysis for YouTube Videos through Transcriptions

## Package Installation
<code>pip install requirements_pip.txt</code>\
    or\
<code>conda install requirements_conda.txt</code>

## Script for storing YouTube Transcriptions 
In <code>transcription_store.sh</code>, please change the
url to the specific video url for which you are trying to
download transcript.<br>
The transcript will get stored in csv format in
<code>emorec_text/data/transcription_storage/<video_id>.csv</code>

## Creating the ground truth data
Given that the transcriptions are stored in <code>emorec_text/data/transcription_storage</code>
and the Deepface emotions are stored in <code>emorec_text/data/video_emotions</code>, the ground truth data
can be created by running the python script <code>python - m emorec_text.code.data_utils.combine_data.py</code>.

## Running baseline
The baseline used for the project is Text2Emotion. To get the baseline results,
you need to run the python script <code>python -m emorec_text.code.models.text_emotion_recognition.py</code>.