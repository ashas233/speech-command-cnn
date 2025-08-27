Speech Command Classification 

A deep learning project that classifies simple speech commands (**yes, no, go, stop**) using **Convolutional Neural Networks (CNNs)** on mel-spectrogram features.  

Features
- Preprocessing with mel-spectrogram extraction  
- CNN-based classifier with dropout regularization  
- Training & evaluation pipeline  
- Model performance visualization  
- Google Colab friendly  
 Dataset
- Custom dataset with **400 audio samples** (100 per class)  
- Pre-extracted mel-spectrograms: shape `(40, 50)`  
- Stored in `mini_speech_dataset.npz`  

 Model Architecture
- **Conv2D + MaxPooling + Dropout** layers  
- Dense layer (128 units, ReLU)  
- Output: Softmax (4 classes)  

 Installation

pip install tensorflow librosa matplotlib numpy scikit-learn
