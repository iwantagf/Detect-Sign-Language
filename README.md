# Detect Sign Language
This project is a deep learning model that can detect sign language from video. Inspired by the second problem in 2025 Northern Olympic AI Challenge.

### Dataset
The dataset is from the 2025 Northern Olympic AI Challenge. It contains 100 classes of sign language videos. You can download from [here](https://drive.google.com/drive/folders/1X3xTeaI8keWD4c-cePjN-xboJGOgQ66A?usp=sharing).

### Data Preprocessing
We fixed a sampling rate of 16 FPS for all videos. Then, we split the dataset into training and validation sets with a ratio of $80:20$. If a video has less than $16$ frames, we will drop it, otherwise, we will randomly sample $16$ frames from it.

### Model
The model uses Vision Transformer (ViT) as backbone to extract features from video frames. Then, it uses a Transformer encoder along with Positional Encoding and Attention Pooling to classify the sign language.

### Training
Training with AdamW optimizer and Weighted Cross Entropy Loss in which labels with less samples have higher weights. The model is trained for $35$ epochs with batch size $32$ and learning rate $10^{-4}$ and macro-F1 for validation. This metric is good for imbalanced dataset where every label has equal weight. You can see that the distribution of the given dataset has a long tail. You can see it by running `python distribution.py`. 

F1-macro on validation set is about $0.9218$ within 1 hour on G4 GPU.


### Augmentation
To prevent overfitting, we use the following augmentation techniques:
- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation
- Random Crop
- Random Color Jitter
- Random Erasing


### Installation
Clone this repository:
```bash
git clone https://github.com/iwantagf/Detect-Sign-Language.git
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

```bash
python main.py
```
