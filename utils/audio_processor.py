import yt_dlp
from pydub import AudioSegment
import os
import shutil

# Create a folder where you want to store all the downloaded files
DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# WHY THIS CHANGED:
# The old code hardcoded a Windows-only ffmpeg path
# (C:\Users\mail2\...\ffmpeg-8.1.2-full_build\bin). That only exists on your
# dev machine -- on any other machine (a server, a teammate's laptop, a
# deployed container) yt-dlp silently can't find ffmpeg, or worse, points
# at a path that doesn't exist and errors confusingly.
#
# Fix: use an env var if you want to override it, otherwise let yt-dlp /
# pydub find ffmpeg on the system PATH (which is how it should be installed
# in any real deployment -- `apt install ffmpeg` on Linux, brew on Mac).
FFMPEG_LOCATION = os.getenv("FFMPEG_LOCATION") or shutil.which("ffmpeg")
if FFMPEG_LOCATION and os.path.isfile(FFMPEG_LOCATION):
    # yt-dlp wants the containing folder, not the exe itself
    FFMPEG_LOCATION = os.path.dirname(FFMPEG_LOCATION)


def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "ffmpeg_location": FFMPEG_LOCATION,  # None is fine -- yt-dlp then uses PATH
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192"
            }
        ],
        # WHY: the previous version had no timeout/retry settings, so a
        # single slow/dropped connection to a googlevideo.com CDN node
        # (very common -- YouTube's CDN throttles or drops long-lived
        # connections) killed the whole pipeline instantly with a raw
        # "Read timed out" traceback and no retry attempt.
        "socket_timeout": 30,        # give up on a stalled connection faster...
        "retries": 5,                # ...but retry several times before failing
        "fragment_retries": 5,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Check duration BEFORE downloading -- long videos are the #1 cause
        # of "it's stuck / taking forever" complaints, since Whisper on CPU
        # runs roughly at or slower than real-time. Warn early instead of
        # letting the user wait 20 minutes to find out.
        info = ydl.extract_info(url, download=False)
        duration_min = (info.get("duration") or 0) / 60
        if duration_min > 15:
            print(f"⚠️  Video is {duration_min:.1f} min long. Whisper "
                  f"transcription on CPU can take as long or longer than "
                  f"the video itself. Consider a shorter clip for testing.")

        info = ydl.extract_info(url, download=True)

        base_filename = ydl.prepare_filename(info)
        filename, _ = os.path.splitext(base_filename)
        wav_filename = filename + ".wav"

        return wav_filename


def convert_to_wav(input_path: str) -> str:
    """Convert any audio and video file using pydub."""
    output_path = os.path.splitext(input_path)[0] + "-converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)  # 16kHz: Whisper's native rate
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 5) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    base_path, ext = os.path.splitext(wav_path)

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{base_path}_chunk_{i}{ext}"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    # NOTE: original code had a copy-paste bug here too:
    #   if source.startswith("https://") or source.startswith("https://"):
    # both branches checked the same prefix, so plain "http://" links (no s)
    # would silently fall through to the "local file" branch and fail.
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube/web URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to wav...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks