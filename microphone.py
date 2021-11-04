#activate microphone and start recording to buffer in chunks
# %%
#
def start_recording():
    import pyaudio
    import wave
    import numpy as np
    import time
    import os
    import sys
    import threading
    import queue
    import icecream as ic

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    fs = 44100  # Record at 44100 samples per second
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"
    q = queue.Queue()

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []
    
    # Store data in chunks for 3 seconds

    # for i in range(0, int(fs / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK * RECORD_SECONDS)
    # ic.ic(data)
    df = np.frombuffer(data, dtype='int16')
    ic.ic(df)
    frames.append(data)
    stream.stop_stream()
    stream.close()
    # break
    #save each chunk from buffer to file

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording')
def save_chunk(data, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()
    
    
    



# %%
start_recording()
# %%