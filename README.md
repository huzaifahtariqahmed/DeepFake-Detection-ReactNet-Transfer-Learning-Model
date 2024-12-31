# DeepFake Image Detection Using ReactNet Based Transfer Learning Classification Model on Celeb DF Dataset

## Transfer Learning for DeepFake Detection
Transfer learning is a powerful machine learning technique where a model pre-trained on a large-scale dataset is fine-tuned for a specific task. In the context of DeepFake detection, transfer learning provides a robust way to leverage the representational power of models trained on massive datasets like ImageNet, enabling improved generalization even with limited domain-specific data.

In this project, we employ **ReActNet** [\[Liu et al., 2020\]](https://arxiv.org/abs/2003.03488), a state-of-the-art Binary Neural Network (BNN) pre-trained on the ImageNet dataset. ReActNet’s binary nature ensures computational efficiency while maintaining high accuracy, making it suitable for resource-constrained devices and real-time DeepFake detection. 

By fine-tuning ReActNet on the **Celeb-DF (v2)** dataset [\[Li et al., 2020\]](https://arxiv.org/abs/1909.12962), we adapt the model to recognize subtle manipulations inherent in DeepFake videos. This approach combines accuracy, computational efficiency, and scalability, making it well-suited for real-world applications in DeepFake detection.

---

## Dataset Preparation 

Our Jupyter Notebook for this can be found [here](https://github.com/huzaifahtariqahmed/DeepFake-Detection-ReactNet-Transfer-Learning-Model/blob/main/Dataset%20preparation%20and%20preprocessing.ipynb).

### Dataset Overview
The dataset used for this study is the **Celeb-DF (v2)** dataset, a widely recognized benchmark for DeepFake detection. It consists of both real and fake videos collected from various sources:

- **Celeb-real:** Authentic videos featuring celebrities.
- **Celeb-synthesis:** Corresponding DeepFake videos generated using advanced synthesis techniques.
- **YouTube-real:** Additional real videos sourced from YouTube.

A provided file, `List_of_testing_videos.txt`, specifies the videos designated for testing.

### Dataset Preparation Steps

#### 1. Dataset Organization
The dataset was split into training and testing sets to ensure a clear separation of data:
```plaintext
/data/
   train/
      real/       <-- Real training videos
      fake/       <-- Fake training videos
   test/
      real/       <-- Real testing videos
      fake/       <-- Fake testing videos
```

#### Frame Extraction

Deep learning models operate on images rather than video streams. Videos were converted into frames, extracting one frame every 10 frames using the OpenCV library. Extracted frames were saved under subfolders corresponding to their source videos:
```plaintext
/data/frames/
   train/
      real/
         video_001/   <-- Extracted frames
         video_002/
      fake/
         video_101/
   test/
      real/
      fake/
```

#### Flattening Frame Folders

To align with the PyTorch [ImageFolder] format, frames were consolidated into flat folders for each class:
```plaintext
/data/frames_flattened/
   train/
      real/   <-- All real frames
      fake/   <-- All fake frames
   test/
      real/
      fake/
```

#### Dataset Reduction

To optimize for limited GPU resources, random sampling was performed:

Training Set: 5,000 real frames and 5,000 fake frames.
Testing Set: 1,000 real frames and 1,000 fake frames.
The reduced dataset structure:

```plaintext
/data/frames_reduced/
   train/
      real/   <-- 5,000 real frames
      fake/   <-- 5,000 fake frames
   test/
      real/   <-- 1,000 real frames
      fake/   <-- 1,000 fake frames
```

#### Verification and Quality Checks

To ensure dataset integrity:

Frame Counts: Verified before and after reduction.
Folder Integrity: Confirmed expected number of frames.
Balanced Distribution: Validated equal representation of real and fake classes.
Final frame counts:

Training Set: 5,000 real frames, 5,000 fake frames.
Testing Set: 1,000 real frames, 1,000 fake frames.

#### Verification and Quality Checks

To ensure dataset integrity:

Frame Counts: Verified before and after reduction.
Folder Integrity: Confirmed expected number of frames.
Balanced Distribution: Validated equal representation of real and fake classes.
Final frame counts:

Training Set: 5,000 real frames, 5,000 fake frames.
Testing Set: 1,000 real frames, 1,000 fake frames.

### Tools and Technologies

The following tools and technologies were used for dataset preparation:

- Python: Programming language for automation and preprocessing.
- OpenCV: For video-to-frame conversion.
- OS & Shutil Libraries: For file and directory management.
- Random Library: For balanced random sampling.
- PyTorch: Ensured compatibility for model training.

## ReActNet-MobileNet Architecture

The **ReActNet-MobileNet** model leverages **MobileNetV2** as a lightweight backbone for feature extraction, making it well-suited for mobile and embedded systems. By incorporating efficient architectural choices, this model significantly reduces computational costs while maintaining high accuracy, making it ideal for resource-constrained environments.

---

### MobileNetV2 Backbone
**MobileNetV2** is a convolutional neural network (CNN) designed for efficient feature extraction. Key features:
- Uses **depthwise separable convolutions** to reduce computational overhead while preserving accuracy.
- Pre-trained on the **ImageNet** dataset for robust feature learning.
- Fine-tuned for the DeepFake detection task by replacing the last fully connected layer for binary classification (real vs fake).  
[Learn more about MobileNetV2 here](https://arxiv.org/abs/1801.04381).

---

### Generalized Activation
To enhance the precision of binary neural networks (BNNs), ReActNet incorporates **generalized activation functions**:
- **RSign (Relaxed Sign):** Allows explicit learning of the distribution reshape and shift at minimal computational cost.
- **RPReLU (Relaxed Parametric ReLU):** Improves non-linear transformations for better representation.  

This combination improves performance by enabling accurate predictions with binarized weights and activations.  
[Learn more in Liu et al., 2020](https://arxiv.org/abs/2003.03488).

---

### Binary Quantization
Binary quantization is a core feature of ReActNet, applied to both weights and activations:
- **Advantages:**
  - Reduces model size significantly.
  - Lowers computational complexity.
  - Enables real-time processing.
- This quantization ensures efficiency while maintaining high accuracy in distinguishing real vs fake media.

---

### Final Classification Layer
The classification process involves:
1. Extracted features from MobileNetV2 and activation layers.
2. A fully connected layer to classify media into two classes:
   - **Real**
   - **Fake**
3. A **softmax function** at the final layer to compute probabilities for each class.

---

### Efficiency and Performance
ReActNet-MobileNet achieves an optimal balance between **accuracy** and **efficiency**:
- **Lower Floating Point Operations (FLOPs):** Enables deployment on mobile and embedded devices.
- Trained for high classification accuracy while maintaining low computational overhead.

#### Key Highlights:
- Lightweight model architecture suitable for real-time DeepFake detection.
- Efficient performance on resource-constrained devices without compromising accuracy.

---

## Experimental Setup

The experimental setup outlines the training process and evaluation metrics used to fine-tune and validate the deepfake detection model.

---

### Training Process

The training process was designed to assess the model's ability to detect deepfakes effectively across varying epochs and conditions. Key details include:

- **Initial Training Phase:**
  - **Optimizer:** Adam
  - **Learning Rate:** 0.001
  - **Batch Size:** 32
  - **Epochs:** 5, 15, 30, and 50
  - Forward and backward passes were iteratively applied to optimize weights, minimizing cross-entropy loss.
  - **Monitoring:** Training and validation accuracies and losses were recorded at each epoch to analyze learning patterns and detect potential overfitting.

- **Fine-Tuning Phase:**
  - **Learning Rate:** Reduced to 0.0001.
  - **Early Stopping:** Implemented to halt training when signs of overfitting were detected.
  - Results from fine-tuning were compared with those from the initial training phase for performance benchmarking.

---

### Evaluation Metrics

A comprehensive set of evaluation metrics was used to measure and understand the model's performance:

- **Accuracy:** Percentage of correctly identified real and fake instances in training, validation, and test datasets.
- **Loss:** Measures the discrepancy between model predictions and actual labels, offering insights into optimization efficiency.
- **Test Accuracy:** Assesses the model's generalization capability to unseen data, highlighting overfitting or underfitting tendencies.
- **Sensitivity and Specificity:** While not explicitly calculated, validation and test accuracy patterns provided indirect insights into the balance between true positives and true negatives.

#### Metric Monitoring:
- **Validation Metrics:** Computed after each epoch to evaluate the model’s ability to generalize beyond the training data.
- **Test Metrics:** Recorded after the completion of each training phase for a robust evaluation of model performance.

This combination of metrics ensures a thorough understanding of the model's strengths and limitations under various configurations, enabling a refined approach to deepfake detection.

---

## Contributors
This project is developed and maintained by:
- [Huzaifah Tariq Ahmed](https://github.com/huzaifahtariqahmed)
- [Daniyal Rahim Areshia](https://github.com/Daniyal-R-A)
- [Aquib Ansari](https://github.com/aqib420)
