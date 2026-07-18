import wave as w

# Create audio file wave object
good_morning = w.open("good_morning.wav" , 'r')

# Read all frames from wave object 
signal_gm = good_morning.readframes(-1)

# View first 10
print(signal_gm[:10])
###################################
import numpy as np

# Open good morning sound wave and read frames as bytes
good_morning = wave.open("good_morning.wav", 'r')
signal_gm = good_morning.readframes(-1)

# Convert good morning audio bytes to integers
soundwave_gm = np.frombuffer(signal_gm , dtype = "int16")

# View the first 10 sound wave values
print(soundwave_gm[:10])
##########################################################
# Read in sound wave and convert from bytes to integers
good_morning = wave.open('good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# Get the sound wave frame rate
framerate_gm = good_morning.getframerate()

# Find the sound wave timestamps
time_gm = np.linspace(start = 0 , stop = len(soundwave_gm) / framerate_gm , num = len(soundwave_gm))

# Print the first 10 timestamps
print(time_gm[:10])