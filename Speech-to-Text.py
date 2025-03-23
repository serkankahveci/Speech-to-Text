import argparse
import os
import sys
import numpy as np
import whisper
import warnings
import subprocess
from scipy import signal
import platform

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def install_dependencies():
    """Install required Python packages"""
    print("Checking and installing required packages...")
    packages = [
        "numpy",
        "scipy",
        "openai-whisper",
        "soundfile",
        "librosa",
        "ffmpeg-python"
    ]
    
    for package in packages:
        try:
            if package == "openai-whisper":
                # Check if whisper is installed
                import whisper
                print(f"✓ whisper is already installed")
            else:
                # Try to import the package
                __import__(package.replace("-", "_"))
                print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Check for ffmpeg
    try:
        subprocess.check_call(
            ["where", "ffmpeg"] if platform.system() == "Windows" else ["which", "ffmpeg"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        print("✓ FFmpeg is installed and available in PATH")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("\n⚠️ WARNING: FFmpeg not found in your system PATH")
        print("Audio processing may not work correctly without FFmpeg")
        
        if platform.system() == "Windows":
            print("\nTo install FFmpeg on Windows:")
            print("1. Download from https://ffmpeg.org/download.html#build-windows")
            print("2. Extract the ZIP file to a location like C:\\ffmpeg")
            print("3. Add the bin folder (e.g., C:\\ffmpeg\\bin) to your PATH environment variable")
            print("4. Restart your command prompt")
        else:
            print("\nTo install FFmpeg on Linux/Mac:")
            print("- Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("- Mac: brew install ffmpeg")
    
    print("\nAll required Python packages are installed.")

def direct_whisper_transcribe(audio_file, model_size="base", language=None):
    """
    Transcribe directly using Whisper without preprocessing
    This is a fallback method when audio loading fails
    """
    print("\nAttempting direct transcription with Whisper...")
    model = whisper.load_model(model_size)
    
    try:
        # Try direct transcription with the file path
        result = model.transcribe(
            audio_file,
            language=language if language else None,
            task="transcribe"
        )
        return result
    except Exception as e:
        print(f"Direct transcription failed: {str(e)}")
        raise

def load_audio(file_path, sample_rate=16000):
    """
    Try multiple methods to load audio
    """
    print(f"Attempting to load audio from {file_path}")
    
    # Method 1: Try with soundfile (best for WAV)
    try:
        import soundfile as sf
        print("Trying soundfile...")
        audio_data, orig_sr = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if orig_sr != sample_rate:
            print(f"Resampling from {orig_sr}Hz to {sample_rate}Hz")
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=sample_rate)
        
        print("Successfully loaded audio with soundfile")
        return audio_data, sample_rate
    except Exception as e:
        print(f"Soundfile loading failed: {str(e)}")
    
    # Method 2: Try with librosa
    try:
        import librosa
        print("Trying librosa...")
        audio_data, _ = librosa.load(file_path, sr=sample_rate, mono=True)
        print("Successfully loaded audio with librosa")
        return audio_data, sample_rate
    except Exception as e:
        print(f"Librosa loading failed: {str(e)}")
    
    # If all methods fail, raise exception
    raise Exception("Failed to load audio with all available methods")

def reduce_noise(audio_data, sample_rate, noise_reduction_strength=0.5):
    """Apply basic noise reduction to audio data"""
    # Ensure audio is normalized but avoid division by zero
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    
    # Apply a high-pass filter (reduce background hum)
    b, a = signal.butter(5, 80/(sample_rate/2), 'highpass')
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    # Get noise profile from first 1 second
    noise_len = min(sample_rate, len(filtered_audio))
    noise_profile = filtered_audio[:noise_len]
    noise_power = np.mean(noise_profile**2)
    
    # Apply spectral subtraction
    filtered_audio_power = filtered_audio**2
    mask = (filtered_audio_power - noise_reduction_strength * noise_power) > 0
    processed_audio = filtered_audio * mask
    
    return processed_audio

def identify_speakers(audio_data, sample_rate, num_speakers=2):
    """Simple placeholder for speaker identification"""
    segments = []
    duration = len(audio_data) / sample_rate
    segment_duration = duration / (num_speakers * 2)  # Alternate between speakers
    
    current_speaker = 0
    start_time = 0
    
    while start_time < duration:
        end_time = min(start_time + segment_duration, duration)
        segments.append({
            'speaker': f"Speaker {current_speaker}",
            'start': start_time,
            'end': end_time
        })
        
        start_time = end_time
        current_speaker = (current_speaker + 1) % num_speakers
    
    return segments

def format_transcript(result, include_speakers=False):
    """Format the transcript result into readable text"""
    if include_speakers:
        formatted_text = ""
        current_speaker = None
        
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown Speaker")
            
            # Add speaker tag when speaker changes
            if speaker != current_speaker:
                formatted_text += f"\n\n{speaker}:\n"
                current_speaker = speaker
            
            formatted_text += segment["text"].strip() + " "
    else:
        # Just concatenate all text segments
        formatted_text = result["text"]
    
    return formatted_text

def transcribe_audio(audio_file, model_size="base", language=None, 
                    noise_reduction=False, detect_speakers=False, num_speakers=2):
    """Main transcription function with fallbacks"""
    print(f"\n--- Starting transcription with Whisper {model_size} model ---")
    
    # Load the Whisper model
    model = whisper.load_model(model_size)
    print(f"✓ Loaded Whisper {model_size} model")
    
    try:
        # Step 1: Try to load and preprocess the audio
        try:
            audio_data, sample_rate = load_audio(audio_file)
            
            if noise_reduction:
                print("Applying noise reduction...")
                audio_data = reduce_noise(audio_data, sample_rate)
                
            audio_for_transcription = audio_data
            loading_succeeded = True
        except Exception as e:
            print(f"Audio preprocessing failed: {str(e)}")
            print("Will attempt direct transcription instead...")
            loading_succeeded = False
        
        # Step 2: Perform transcription
        if loading_succeeded:
            # Transcribe with preprocessed audio
            print(f"Transcribing audio{' (language: ' + language + ')' if language else ''}...")
            result = model.transcribe(
                audio_for_transcription,
                language=language if language else None,
                task="transcribe"
            )
        else:
            # Direct transcription with file path as fallback
            result = direct_whisper_transcribe(audio_file, model_size, language)
        
        # Step 3: Add speaker identification if requested and possible
        if detect_speakers and loading_succeeded:
            print(f"Detecting speakers (estimated: {num_speakers})...")
            speaker_segments = identify_speakers(audio_data, sample_rate, num_speakers)
            
            for segment in result["segments"]:
                for speaker_segment in speaker_segments:
                    if (segment["start"] >= speaker_segment["start"] and 
                        segment["start"] < speaker_segment["end"]):
                        segment["speaker"] = speaker_segment["speaker"]
                        break
                else:
                    segment["speaker"] = "Unknown Speaker"
        
        return result
    
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        if "cuda" in str(e).lower() or "gpu" in str(e).lower() or "memory" in str(e).lower():
            print("This may be a GPU memory issue. Try using the 'tiny' model.")
        raise

def main():
    # Create parser with two subcommands
    parser = argparse.ArgumentParser(description="Speech-to-Text with OpenAI Whisper")
    subparsers = parser.add_subparsers(dest="command")
    
    # Setup command for installing dependencies
    setup_parser = subparsers.add_parser("setup", help="Install required dependencies")
    
    # Transcribe command for actual transcription
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio to text")
    transcribe_parser.add_argument("audio_file", help="Path to audio file")
    transcribe_parser.add_argument("--model", default="base", 
                                  choices=["tiny", "base", "small", "medium", "large"],
                                  help="Whisper model size")
    transcribe_parser.add_argument("--language", help="Language code (e.g., 'en', 'fr')")
    transcribe_parser.add_argument("--noise-reduction", action="store_true", 
                                  help="Apply noise reduction")
    transcribe_parser.add_argument("--detect-speakers", action="store_true", 
                                  help="Detect different speakers")
    transcribe_parser.add_argument("--num-speakers", type=int, default=2, 
                                  help="Number of speakers to detect")
    transcribe_parser.add_argument("--output", help="Output text file path")
    
    args = parser.parse_args()
    
    # Handle setup command
    if args.command == "setup":
        install_dependencies()
        print("\n✓ Setup complete. You can now use the 'transcribe' command to convert speech to text.")
        return
    
    # Handle transcribe command
    if args.command == "transcribe":
        audio_file = args.audio_file
        model = args.model
        language = args.language
        noise_reduction = args.noise_reduction
        detect_speakers = args.detect_speakers
        num_speakers = args.num_speakers
        output = args.output
    else:
        parser.print_help()
        print("\n\nExamples:")
        print("  Install dependencies:  python Speech-to-Text.py setup")
        print("  Transcribe audio:      python Speech-to-Text.py transcribe audio.mp3 --model base --language en")
        return
    
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found.")
        return
    
    try:
        # Perform transcription
        result = transcribe_audio(
            audio_file,
            model_size=model,
            language=language,
            noise_reduction=noise_reduction,
            detect_speakers=detect_speakers,
            num_speakers=num_speakers
        )
        
        # Format the transcript
        formatted_transcript = format_transcript(result, include_speakers=detect_speakers)
        
        # Output result
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(formatted_transcript)
            print(f"\n✓ Transcript saved to {output}")
        else:
            print("\nTranscription Result:")
            print("-" * 50)
            print(formatted_transcript)
            print("-" * 50)
            
        print("\n✓ Transcription complete!")
        
    except Exception as e:
        print(f"\n❌ Transcription failed: {str(e)}")
        print("\nTroubleshooting Tips:")
        print("1. Run 'python Speech-to-Text.py setup' to install dependencies")
        print("2. Install FFmpeg (see instructions above)")
        print("3. Try using a different audio file format (MP3, WAV, etc.)")
        print("4. Try the 'tiny' model: --model tiny")
        print("5. Check if your audio file contains actual speech")
        print("6. Make sure your network connection is stable for model downloads")

if __name__ == "__main__":
    main()