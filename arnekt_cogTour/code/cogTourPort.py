# -*- coding: utf-8 -*-

# importing required libraries
import os
import io
import config
import shutil
import subprocess
import jsonpickle
import scipy.sparse
import pandas as pd
from tqdm import tqdm

from google.cloud import vision
import speech_recognition as sr
from google.cloud import translate
from google.cloud.vision import types

from sklearn.externals import joblib
from flask import Flask, request, Response
from scipy.spatial.distance import euclidean

# configuring credentials
##GOOGLE_CLOUD_SPEECH_CREDENTIALS = config.GOOGLE_CLOUD_SPEECH_CREDENTIALS
with open("../config/BarathiGaneshHB-8f812f6acdf1.json") as f:
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = f.read()

faq_vocabulary = joblib.load(config.faq_voabulary)
faq_mat = scipy.sparse.load_npz(config.faq_mat)
faq_data = pd.read_csv(config.faq_data, header=None)

# get faq answers
def get_faq_ans(temp_text):
    
    test_vec = faq_vocabulary.transform([temp_text])
    distance_sim = list()
    for i in range(0,faq_mat.shape[0]):
        distance_sim.append(euclidean(test_vec.todense().tolist()[0], faq_mat[i,:].todense().tolist()[0]))
    faq_ans = list(faq_data[2])[distance_sim.index(min(distance_sim))]
    
    return faq_ans
# function for performing optical character recognition (image->text)
def get_ocr(temp_image_file, temp_source):
    
    client = vision.ImageAnnotatorClient()
    with io.open(temp_image_file, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    image_context = types.ImageContext(language_hints=[temp_source])
    response = client.text_detection(image=image, image_context=image_context)
    texts = response.text_annotations
    source_text = []
    for text in texts:
        text.description = text.description.strip()
        if text.description in source_text:
            pass
        else:
            source_text.append(text.description)
    source_text = ' '.join(source_text)

    return source_text


# function for performing text translation (hindi->english->tamil) = (source->english->target)
def get_translation(temp_text, temp_target, temp_source):
    
    translate_client = translate.Client()
    if temp_target == temp_source:
        temp_result = temp_text
    elif temp_source != 'en':
        temp_result = (translate_client.translate(temp_text, target_language='en', source_language=temp_source))['translatedText']
        if temp_target != 'en':
            temp_result = translate_client.translate(temp_result, target_language=temp_target, source_language=temp_source)['translatedText']
        
    else:
        temp_result = translate_client.translate(temp_text, target_language=temp_target, source_language=temp_source)['translatedText']
    
    return temp_result

# function for performing text to english translation
def get_translation_en(temp_text, temp_source):
    
    translate_client = translate.Client()
    if temp_source != 'en':
        temp_result = (translate_client.translate(temp_text, target_language='en', source_language=temp_source))['translatedText']
    else:
        temp_result = temp_text
    
    return temp_result


# function to perform speech to text
def get_speech2text(audio_file_path, source_langauge):
    
    # processing audio files (speech -> chunks of speech)
    op_audio_file = audio_file_path
    op_audio_file_chunk = config.op_audio_file_chunk
    chunk_path = config.chunk_path
    if os.path.isdir(chunk_path):
        shutil.rmtree(chunk_path)
        os.makedirs(chunk_path)
    else:
        os.makedirs(chunk_path)
    subprocess.call(['ffmpeg', '-i', op_audio_file,'-f', 'segment', '-segment_time','10', '-c', 'copy', op_audio_file_chunk])
    r = sr.Recognizer()
    files = os.listdir(chunk_path)
    all_text = []     
    for f in tqdm(files):
        name = os.path.join(chunk_path, f)
        with sr.AudioFile(name) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS, language=source_langauge)
        except:
            text = ' '
        all_text.append(text)
    text = ' '.join(all_text)
    
    return text


# Initialize the Flask application
app = Flask(__name__)

# native guide
@app.route('/api/nativeguide', methods=['POST'])
def nativeguide():
    
    source = request.form['source']
    target = request.form['target']
    audio_file = request.files['audio']
    audio_file.save(os.path.join(config.speech_path, 'test.wav'))
    audio_file_path = os.path.join(config.speech_path, 'test.wav')
    # do some fancy processing here....
    text = get_speech2text(audio_file_path, str(source))
    native_text = get_translation(text, target, source)
    
    response = {'message': native_text}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# polyglot
@app.route('/api/polyglot', methods=['POST'])
def polyglot():
    
    source = request.form['source']
    target = request.form['target']
    img_file = request.files['image']
    img_file.save(os.path.join(config.vision_path, 'test.png'))
    img_file_path = os.path.join(config.vision_path, 'test.png')
    # do some fancy processing here....
    temp_text1 = get_ocr(img_file_path, source)
    poly_text = get_translation(temp_text1, target, source)
    response = {'message': poly_text}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# report a problem
@app.route('/api/reportplm', methods=['POST'])
def reportplm():
    
    source = request.form['source']
    target = request.form['target']
    if 'image' in request.files:
        img_file = request.files['image']
        img_file.save(os.path.join(config.vision_path, 'test.png'))
        img_file_path = os.path.join(config.vision_path, 'test.png')
    else:
        pass
    if 'audio' in request.files:
        audio_file = request.files['audio']
        audio_file.save(os.path.join(config.speech_path, 'test.wav'))
        audio_file_path = os.path.join(config.speech_path, 'test.wav')
        # do some fancy processing here....
        print target
        text = get_speech2text(audio_file_path, str(source))
        report_text = get_translation_en(text, source)
        with open(config.report_text_path, "a") as myfile:
            myfile.write(report_text+'\n')
        text = get_translation(report_text, target, source)
    else:
        text = 'Please insert provide the audio or image'
        text = get_translation(text, target, source)    
    response = {'message': (text)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# help your buddy
@app.route('/api/helpurbuddy', methods=['POST'])
def helpurbuddy():
    
    source = request.form['source']
    target = request.form['target']
    audio_file = request.files['audio']
    audio_file.save(os.path.join(config.speech_path, 'test.wav'))
    audio_file_path = os.path.join(config.speech_path, 'test.wav')
    # do some fancy processing here....
    text = get_speech2text(audio_file_path, target)
    govt_text = get_translation_en(text, target)
    with open(config.report_text_path, "a") as myfile:
        myfile.write(govt_text+'\n')
    # do some fancy processing here....
    text = 'We have noted your problem, getting help!'
    text = get_translation(text, target, source)
    response = {'message': text }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

# faq
@app.route('/api/faq', methods=['POST'])
def faq():
    
    source = request.form['source']
    target = request.form['target']
    audio_file = request.files['audio']
    audio_file.save(os.path.join(config.speech_path, 'test.wav'))
    audio_file_path = os.path.join(config.speech_path, 'test.wav')
    # do some fancy processing here....
    text = get_speech2text(audio_file_path, str(target))
    query_text = get_translation_en(text, target)
    faq_ans = get_faq_ans(query_text)
    faq_text = get_translation(faq_ans, target, 'en')
    response = {'message': faq_text}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


app.run(host="localhost", port=8000)
