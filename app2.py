import pylab
import streamlit as st
import numpy as np
import pyaudio
import wave
import librosa
import librosa.display
import PIL.Image
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100


def record_audio(duration):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # wave_file = wave.open(output_name, 'wb')
    # wave_file.setnchannels(CHANNELS)
    # wave_file.setsampwidth(p.get_sample_size(FORMAT))
    # wave_file.setframerate(RATE)
    # wave_file.writeframes(b''.join(frames))
    # wave_file.close()

    return np.frombuffer(b''.join(frames), dtype=np.int16)


p = pyaudio.PyAudio()


def play_audio(data):
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)
    stream.write(data.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


def plot_spectrogram(rec_name):
    y, sr = librosa.load(rec_name)
    fig = plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr,n_mels=256,hop_length=256)
    spec_db = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(spec_db,
                             y_axis='mel', fmax=8000, x_axis='time')
    plt.tight_layout()
    # spec_scaled = (spec_db + 80) / 80  # Scale between 0 and 1
    # spec_resized = np.resize(spec_scaled, (256, 256))  # Resize to match the model input shape
    cur = rec_name.split('/')[1]
    print(cur)
    plt.savefig(f"{cur.split('.')[0]}.png")
    return spec_db,fig

st.title("Audio Calculator")
# model = keras.models.load_model('trained_model.h5')
def predict_word(spect_path):
    model = keras.models.load_model('trained_model.h5')
    image = PIL.Image.open(spect_path)
    resized_image = image.resize((256, 256))
    input_image = tf.image.convert_image_dtype(resized_image, tf.float32)
    input_image = tf.expand_dims(input_image, axis=0)
    prediction = model.predict(input_image)
    max_value = max(prediction)
    predicted_class = [i for i,x in enumerate(prediction) if x == max_value]
    return predicted_class

# Record three times
def rec_3_times():
    global recordings
    for i in range(3):
        st.write(f"Recording {i + 1}...")
        data = record_audio(duration=5)
        st.write("Finished recording.")

        # Save the recording to a WAV file
        filename = f"recording_{i + 1}.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(data.tobytes())

        # Append the recording to the list
        recordings.append(data)

        # Add a button to play the recording

if "load_state" not in st.session_state:
     st.session_state.load_state = False

if st.button("Execute Function") and not st.session_state.load_state:
        recordings = []

        # Call the function
#         for i in range(3):
#             st.write(f"Recording {i + 1}...")
#             data = record_audio(duration=5)
#             st.write("Finished recording.")

#             # Save the recording to a WAV file
#             filename = f"recording_{i + 1}.wav"
#             with wave.open(filename, 'wb') as wf:
#                 wf.setnchannels(CHANNELS)
#                 wf.setsampwidth(p.get_sample_size(FORMAT))
#                 wf.setframerate(RATE)
#                 wf.writeframes(data.tobytes())

#             # Append the recording to the list
#             recordings.append(data)

#             # Add a button to play the recording
#             if st.button(f"record again {i + 1}"):
#                 play_audio(data)
        st.session_state.load_state = True

c1, c2, c3 = st.columns(3)
# if st.button("Execute Function") and not st.session_state.load_state:
#     pass
# Record the first number
if "load_state" in st.session_state and st.session_state.load_state:
    with c1:
        st.header("Play the first digit")
        data, sr = librosa.load('./recording_1.wav')
        st.audio(data, sample_rate=RATE)
        if st.button('Present spectrogram 1'):
            spectrogram,fig = plot_spectrogram('./recording_1.wav')
            st.pyplot(fig)
            print(predict_word('./recording_1.png'))
            #st.write(predict_word(spectrogram))

    # Record the operation
    with c2:
        st.header("Play the operation (+, -, *, /)")
        data, sr = librosa.load('./recording_1.wav')
        st.audio(data, sample_rate=RATE)
        if st.button('Present spectrogram 2'):
            plot_spectrogram('./recording_2.wav')

            st.pyplot()
            #st.write(predict_word(spectrogram))
    # Record the second number
    with c3:
        st.header("Play the second digit")
        data, sr = librosa.load('./recording_1.wav')
        st.audio(data, sample_rate=RATE)
        if st.button('Present spectrogram 3'):
            plot_spectrogram('recording_3.wav')
            st.pyplot()
            #st.write(predict_word(spectrogram))


    # Play back the recordings
    if st.button("Play back recordings"):
        data1, sr = librosa.load('./recording_1.wav')
        data2, sr = librosa.load('./recording_2.wav')
        data3, sr = librosa.load('./recording_3.wav')
        play_audio(data1)
        play_audio(data2)
        play_audio(data3)
