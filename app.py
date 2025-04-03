import cv2
import numpy as np
import time
from scipy.fftpack import fft
import scipy.signal as signal

# Initialize camera with DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

fps = 30  # Adjust FPS based on your camera
window_size = 300  # Number of frames for FFT analysis
buffer = []  # To store signal values

def extract_signal(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[100:200, 150:250]  # Define a region of interest
    avg_intensity = np.mean(roi)
    return avg_intensity

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    signal_value = extract_signal(frame)
    buffer.append(signal_value)
    
    if len(buffer) > window_size:
        buffer.pop(0)
        
        # Apply Gaussian filtering for subtle changes
        smoothed_signal = signal.savgol_filter(buffer, 51, 3)
        
        # Apply FFT
        signal_fft = fft(smoothed_signal)
        freqs = np.fft.fftfreq(len(smoothed_signal), d=1/fps)
        positive_freqs = freqs[freqs > 0]
        magnitudes = np.abs(signal_fft[:len(positive_freqs)])
        
        # Apply Bandpass Filtering (1-2 Hz for heart rate)
        low, high = 1, 2  # Hz
        valid_indices = (positive_freqs >= low) & (positive_freqs <= high)
        filtered_freqs = positive_freqs[valid_indices]
        filtered_magnitudes = magnitudes[valid_indices]
        
        # Detect peak frequency
        if len(filtered_freqs) > 0:
            peak_freq = filtered_freqs[np.argmax(filtered_magnitudes)]
            bpm = peak_freq * 60
            cv2.putText(frame, f'BPM: {int(bpm)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
