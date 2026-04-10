'''
@file      audio_script.py
@brief     script for Whisper Speech to Text with Mems mic
@author    Shyam Jha (Avinashee Tech)
'''

import serial
import wave
import time
import whisper

# # --- Serial Configuration ---
SERIAL_PORT = 'COM8'  # Replace with your serial port 
BAUD_RATE = 921600
OUTPUT_FILENAME = 'recorded_audio.wav'

# --- Audio characteristics ---
SAMPLE_RATE = 16000   # Samples per second
CHANNELS = 1          # Mono
SAMPLE_WIDTH = 4      # 4 bytes = 32-bit audio
RECORD_SECONDS = 20   # Duration to record

# --- Whisper configuration ---
TRANSCRIBE = True

'''
@brief  connect to uart device, record i2s audio data and saves it to a WAV file.
@retval None
'''
def record_audio_from_serial():

    audio_frames = [] # List to save audio data

    try:
        # Establish serial connection with a timeout of 1 second
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1
        )
        print(f"Connected to serial port {SERIAL_PORT} at {BAUD_RATE} baud.")

        # Wait a moment for the connection to establish and DMA to start
        time.sleep(2) 

        print(f"Recording for {RECORD_SECONDS} seconds...")
        start_time = time.time()

        while (time.time() - start_time) < RECORD_SECONDS:
            # Read all available bytes from the serial buffer
            data = ser.read(ser.in_waiting)
            if data:
                audio_frames.append(data)
        
        # Stop recording and close the serial port
        ser.close()
        print("Recording stopped. Serial port closed.")

    except serial.SerialException as e:
        print(f"Error opening or communicating with the serial port: {e}")
        return

    if not audio_frames:
        print("No audio data received.")
        return

    # Combine the list of byte objects into a single byte string
    pcm_data = b''.join(audio_frames)
    print(f"Received {len(pcm_data)} bytes of PCM data.")

    # Save the PCM data to a WAV file
    try:
        with wave.open(OUTPUT_FILENAME, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm_data)
        print(f"Audio saved successfully to {OUTPUT_FILENAME}")

        time.sleep(1)  # some delay
        audio_decode() # decode the recorded audio


    except Exception as e:
        print(f"Error writing WAV file: {e}")


'''
@brief  decode the recorded audio using Whisper model
@retval None
'''
def audio_decode():

    # load the whisper model
    model = whisper.load_model("small")
    if TRANSCRIBE:
        result = model.transcribe(OUTPUT_FILENAME, temperature=0, verbose=False, task="translate")
        print(result["text"])
    else:
        audio = whisper.load_audio(OUTPUT_FILENAME) # load audio
        audio = whisper.pad_or_trim(audio)   # pad/trim to fit 30 seconds 
        
        # make log-mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio,n_mels = model.dims.n_mels).to(model.device)
        
        # detect the spoken language
        _,probs = model.detect_language(mel)
        for(lang,percent) in probs.items():
            if(percent*100 > 2):
                print(f"lang:{lang}, prob:{percent*100:.2f}%")
        print(f"Detected Language: {max(probs, key=probs.get)}")
        
        # decode the audio
        options = whisper.DecodingOptions(temperature=0.0)
        result = whisper.decode(model, mel, options)
        
        # print the recognized text
        print(result.text)


if __name__ == '__main__':
    record_audio_from_serial()
