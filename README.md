# Speech-to-Text Transcription Tool

## Overview
This project provides a speech-to-text transcription tool using OpenAI's Whisper model. It supports various audio formats and includes features such as noise reduction and speaker identification.

## Features
- Convert audio files into text using Whisper.
- Support for multiple languages.
- Noise reduction for clearer transcription.
- Speaker identification for multi-speaker conversations.
- Output transcription to a text file.

## Installation

### Prerequisites
Ensure you have **Python 3.8+** installed. Then, clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/yourusername/Speech-to-Text.git
cd Speech-to-Text

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Set Up
Before running the tool, install any additional dependencies:
```bash
python Speech-to-Text.py setup
```

### 2. Convert Audio to Text
To transcribe an audio file (e.g., `audio.mp3`), run:
```bash
python Speech-to-Text.py transcribe audio.mp3 --model base --language en
```

#### Available Arguments:
| Argument | Description |
|----------|-------------|
| `audio.mp3` | Path to the audio file |
| `--model` | Model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `--language` | Language code (e.g., `en` for English, `tr` for Turkish) |
| `--noise-reduction` | Enable noise reduction |
| `--detect-speakers` | Enable speaker identification |
| `--num-speakers` | Specify the number of speakers |
| `--output` | Specify output file (e.g., `transcript.txt`) |

### 3. Advanced Example
To transcribe a noisy multi-speaker file and save the output:
```bash
python Speech-to-Text.py transcribe audio.mp3 --model base --language en --noise-reduction --detect-speakers --num-speakers 2 --output transcript.txt
```

## License
This project is licensed under the MIT License.

## Contributing
Feel free to open issues or submit pull requests to improve the project.

## Contact
For questions or suggestions, reach out via GitHub Issues.

