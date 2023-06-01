import pylab
import streamlit as st
import numpy as np
import pyaudio
import wave
import librosa
import librosa.display
from matplotlib import pyplot as plt
from tensorflow import keras
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
    # sound_info, frame_rate =
    # get_wav_info(rec_name)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                             y_axis='mel', fmax=8000, x_axis='time')
    plt.tight_layout()
    # spec_db = librosa.power_to_db(S, ref=np.max)
    # spec_scaled = (spec_db + 80) / 80  # Scale between 0 and 1
    # spec_resized = np.resize(spec_scaled, (256, 256))  # Resize to match the model input shape
    # return spec_resized.reshape(1, 256, 256, 1)

st.title("Audio Calculator")
# def predict_word(spect):
#     model = keras.models.load_model('trained_model.h5')
#     prediction = model.predict(spect)
#     predicted_class = np.argmax(prediction)
#     return predicted_class

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

print(st.session_state.load_state)
if st.button("Execute Function") and not st.session_state.load_state:
        recordings = []

        # Call the function
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
            if st.button(f"record again {i + 1}"):
                play_audio(data)
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
            plot_spectrogram('./recording_1.wav')
            st.pyplot()
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

print(st.session_state.load_state)