# importing required libraries
import os

# configuring credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../config/BarathiGaneshHB-8f812f6acdf1.json"
with open("../config/BarathiGaneshHB-8f812f6acdf1.json") as f:
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = f.read()


# paths
# for audio
op_audio_file_chunk = '../data/speech/parts/out%09d.wav'
chunk_path = '../data/speech/parts'
speech_path = '../data/speech'

# for vision
vision_path = '../data/vision'

# for text
report_text_path = '../data/text/report.plm.txt'


# model paths

faq_voabulary = '../data/faq/vocabulary.pkl'
faq_data = '../data/faq/faq.csv'
faq_mat = '../data/faq/faq_matrix.npz'

