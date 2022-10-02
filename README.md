# TTS with WORLD

- WIP
- 『Pythonで学ぶ音声合成』 Chapter 5 and 6 with TensorFlow

## Development Setup on Apple Silicon Mac
- Install Miniforge. https://developer.apple.com/metal/tensorflow-plugin/
- `conda env create -f=conda_env.yml -n tts-world`
- `conda activate tts-world`
- `python -m pip install -e .`

### Update the env settings file
- `conda env export > conda_env.yml`

## Tips
- If you see `Library not loaded: '@rpath/libcblas.3.dylib'` error, please try `python -m pip install --force-reinstall numpy==1.22.4`.
