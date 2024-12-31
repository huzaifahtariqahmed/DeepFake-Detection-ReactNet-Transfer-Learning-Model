# DeepFake Image Detection Using ReactNet Based Transfer Learning Classification Model on Celeb DF Dataset

## Transfer Learning for DeepFake Detection
Transfer learning is a powerful machine learning technique where a model pre-trained on a large-scale dataset is fine-tuned for a specific task. In the context of DeepFake detection, transfer learning provides a robust way to leverage the representational power of models trained on massive datasets like ImageNet, enabling improved generalization even with limited domain-specific data.

In this project, we employ **ReActNet** [\[Liu et al., 2020\]](https://arxiv.org/abs/2003.03488), a state-of-the-art Binary Neural Network (BNN) pre-trained on the ImageNet dataset. ReActNet’s binary nature ensures computational efficiency while maintaining high accuracy, making it suitable for resource-constrained devices and real-time DeepFake detection. 

By fine-tuning ReActNet on the **Celeb-DF (v2)** dataset [\[Li et al., 2020\]](https://arxiv.org/abs/1909.12962), we adapt the model to recognize subtle manipulations inherent in DeepFake videos. This approach combines accuracy, computational efficiency, and scalability, making it well-suited for real-world applications in DeepFake detection.

---

## Dataset Preparation

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

## Contributors
This project is developed and maintained by:
- [Huzaifah Tariq Ahmed](https://github.com/huzaifahtariqahmed)
- [Daniyal Rahim Areshia](https://github.com/Daniyal-R-A)
- [Aquib Ansari](https://github.com/aqib420)
