# file where llm and user input (mic) get combined and where the outputting happens

import whisper
import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import time
import pyttsx3
import llm_interaction

Model = 'base'  # Whisper model size (tiny, base, small, medium, large)
SampleRate = 44100  # Stream device recording frequency
BlockSize = 35  # Block size in milliseconds
Threshold = 0.01  # Minimum volume threshold to activate listening
Vocals = [75, 750]  # Frequency range to detect sounds that could be speech
EndBlocks = 40  # Number of blocks to wait before sending to Whisper
SilenceDuration = 10  # Duration of silence in seconds to stop listening

interaction = llm_interaction.llmInteraction()

memory_file = 'exact/path/to/memory_file.txt'
output_file = 'exact/path/to/output_file.txt'
transcribed_text_file = 'exact/path/to/transcribed_text_file.txt'

# All available voices on my pc
# HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0
# HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_DE-DE_HEDDA_11.0
# HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_JA-JP_HARUKA_11.0
voice_id = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0'

engine = pyttsx3.init()


class Asuka:
    def __init__(self):
        self.running = True
        self.padding = 0
        self.prevblock = self.buffer = np.zeros((0, 1))
        self.fileready = False
        print("\033[96mLoading Asuka..\033[0m", end='', flush=True)
        self.model = whisper.load_model(f'{Model}')
        print("\033[90m Done.\033[0m")
        self.last_audio_time = time.monotonic()
        self.transcribed_text = ""

        with open(memory_file, 'r', encoding='utf-8') as m:
            lines = m.readlines()
            last_x_lines = lines[-6:]
        print(''.join(last_x_lines))

    def callback(self, indata, frames, time, status):
        if not any(indata):
            print('\033[31m.\033[0m', end='', flush=True)
            return

        freq = np.argmax(
            np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames
        if np.sqrt(np.mean(indata**2)) > Threshold and Vocals[0] <= freq <= Vocals[1]:
            print('.', end='', flush=True)
            if self.padding < 1:
                self.buffer = self.prevblock.copy()
            self.buffer = np.concatenate((self.buffer, indata))
            self.padding = EndBlocks
        else:
            self.padding -= 1
            if self.padding > 1:
                self.buffer = np.concatenate((self.buffer, indata))
            elif self.padding < 1 < self.buffer.shape[0] > SampleRate:
                self.fileready = True
                write('dictate.wav', SampleRate, self.buffer)
                self.buffer = np.zeros((0, 1))
            elif self.padding < 1 < self.buffer.shape[0] < SampleRate:
                self.buffer = np.zeros((0, 1))
                print("\033[2K\033[0G", end='', flush=True)
            else:
                self.prevblock = indata.copy()

    def process(self):
        if self.fileready:
            print("\n\033[90mTranscribing..\033[0m")
            result = self.model.transcribe('dictate.wav', fp16=False)
            print(f"\033[1A\033[2K\033[0G{result['text']}")
            os.remove('dictate.wav')
            self.fileready = False
            # Save the transcribed text to the variable
            self.transcribed_text = result['text']

            edited = 'Me:' + self.transcribed_text + '\n'
            with open(transcribed_text_file, 'w', encoding='utf-8') as a:
                a.write(edited)

            interaction.generate_response()

            with open(output_file, 'r', encoding='utf-8') as b:
                answer = "".join(b.read())

            engine.setProperty('rate', 150)
            engine.setProperty('voice', voice_id)

            engine.say(answer.replace('Asuka: ', ''))
            engine.runAndWait()

    def listen(self):
        while True:
            self.running = True
            self.last_audio_time = time.monotonic()

            with sd.InputStream(callback=self.callback, channels=1, samplerate=SampleRate, blocksize=int(BlockSize*SampleRate/1000)):
                while self.running:
                    self.process()

                    # Check for silence duration to stop listening
                    audio_time = time.monotonic()
                    silence_duration = audio_time - self.last_audio_time
                    if silence_duration >= SilenceDuration:
                        self.running = False

                    self.last_audio_time = audio_time
                    time.sleep(0.1)


def main():
    handler = Asuka()
    handler.listen()


if __name__ == '__main__':
    main()
