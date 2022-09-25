# TTS with WORLD

## Development Setup on Apple Silicon Mac
- Install Miniforge. https://developer.apple.com/metal/tensorflow-plugin/
- `conda create -n tts-world`
- `conda activate tts-world`
- `conda install -c apple tensorflow-deps`
- `python -m pip install tensorflow-macos`
- `conda install black pytest pylint isort mypy`
- `conda env export -n tts-world > conda_env.yml`


WIP.

- 『Pythonで学ぶ音声合成』 Chapter 5 and 6 with TensorFlow
