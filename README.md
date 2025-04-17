# Instructions to follow
### 1. Create a conda environment
Create a new conda environment named [`tf`](https://www.tensorflow.org/api_docs/python/tf) with the following command.
```shell
conda create --name tf python=3.9
```

You can deactivate and activate it with the following commands.
```shell
conda deactivate
conda activate tf
```
### 2. GPU setup
You can skip this section if you only run TensorFlow on CPU.
```shell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```
### 3. Install TensorFlow
Anything above 2.10 is not supported on the GPU on Windows Native
```shell
pip install "tensorflow<2.11" 
```
### 4. Verify the installation
Verify the GPU setup:
```shell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
### 5. Install the following dependencies after activating conda enviroment
```shell
pip install numpy==1.23.5 
pip install matplotlib==3.6.0 
pip install pillow==9.2.0 
pip install scikit-learn==1.1.3
pip install opencv-python tensorflow spotipy
```

## How to Set Up the Emotion-Based Music Player

To get this program working, you'll need to:

1. **Set up Spotify API access**:
    - Create a Spotify Developer account at [developer.spotify.com](https://developer.spotify.com/)
    - Create a new application in the developer dashboard
    - Get your Client ID and Client Secret
    - Set the Redirect URI to `http://localhost:8888/callback`
    - Replace the placeholders in the code with your actual credentials
2. **Customize the playlists**:
	- The code includes some default Spotify playlist URIs for each emotion
	- You can customize these with your own preferred playlists
	- Get playlist URIs from Spotify by right-clicking a playlist and selecting "Share > Copy Spotify URI"

## This is Streamlit version of the Project. We have created a website for this [Here](https://moodtune.surge.sh/)
