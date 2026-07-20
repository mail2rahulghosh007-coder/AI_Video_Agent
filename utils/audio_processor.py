import yt_dlp
from pydub import AudioSegment
import os

# Create a folder where you want to store all the downloaded files
DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        # Direct path to the bin folder containing ffmpeg.exe and ffprobe.exe
        "ffmpeg_location": r"C:\Users\mail2\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin", 
        "postprocessors": [  
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192"
            }
        ],
        "quiet": False
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        
        # Safely determine the final .wav filename path
        base_filename = ydl.prepare_filename(info)
        filename, _ = os.path.splitext(base_filename)
        wav_filename = filename + ".wav"
        
        return wav_filename



# The above file waas for online video link , now we want create wav file for any video file like mp4 which is created in my local machine 
def convert_to_wav(input_path:str)->str:
    """convert any audio and video file using pydub."""
    output_path=os.path.splitext(input_path)[0]+"-converted.wav"
    audio=AudioSegment.from_file(input_path)
    audio=audio.set_channels(1).set_frame_rate(16000) # using this you can convert frequency to 16khz...because in this freq. whisper is more efficicent
    audio.export(output_path,format="wav")
    return output_path


# now this function is for chunking 
def chunk_audio(wav_path: str, chunk_minutes: int = 5) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000  # chunk milliseconds
    chunks = []
    
    # Separate 'downloads/audio' from '.wav'
    base_path, ext = os.path.splitext(wav_path)
    
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        
        # Generates: downloads/audio_chunk_0.wav
        chunk_path = f"{base_path}_chunk_{i}{ext}"
        
        chunk.export(chunk_path, format="wav") # this is the line which actually saves the chunks in harddrive 
        chunks.append(chunk_path)
        
    return chunks


# this is the function which will connect all the above function 
def process_input(source:str)->list:
    if source.startswith("https://") or source.startswith("https://"):
        print("detecting youtube url. Downloading audio")
        wav_path=download_youtube_audio(source)
    else:
        print("detected loacl video converting to wav")
        wav_path=convert_to_wav(source)

    print("chunking video")
    chunks=chunk_audio(wav_path)
    print(f"Audio ready-{len(chunks)}chunk(s) created.")
    return chunks