#activate microphone and start recording to buffer in chunks
# %%
import pyaudio
import wave
import numpy as np
import time
import os
import sys
import threading
from threading import Thread
import multiprocessing 
from multiprocessing import Process, Queue
import icecream as ic

q = Queue()

def start_recording():
   
    
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    fs = 44100  # Record at 44100 samples per second
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"
    q = Queue()

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
    q.put(df)
    frames.append(data)
    
    save_chunk(df)
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
    
    
    

#show percentage size of buffer in use
def show_buffer_size(queue1):
    while True:
        print(queue1.qsize())
        print(q.get())
        time.sleep(1)

# %%


start_recording()
# %%

process1 = Process(target=start_recording)
process2 = Process(target=show_buffer_size, args=(q,))

process1.start()


# %%
process2.start()
process1.join()
process2.join()
# %%
