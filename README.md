# CCA 3 - Computer Vision Practical Assignment

**Name:** Shaurya Singh  
**PRN:** 1272251359  
**Repository:** https://github.com/DA-Shaurya/CV_assig  

---

## 📁 Repository Structure

- `Shaurya_Singh_CCA3_Computer_Vision.ipynb`: Main interactive technical document containing full algorithm implementations across Units 3, 4, and 5, executed image outputs, measured quantitative tables, visual comparisons, and analytic discussions.
- `Shaurya_Singh_CCA3_Computer_Vision.docx`: Formal technical report document including all embedded high-resolution figures, verified tables, problem formulations, and conclusions.
- `pipeline.py`: Standalone, end-to-end command-line executable Python implementation of all computer vision tasks.
- `requirements.txt`: Python package dependencies.
- `data/originals/`:
  - `photo1.png`: First original student input portrait (outdoor scene under open sky in natural daylight).
  - `photo2.jpg`: Second original student input portrait (indoor banquet hall setting with pillar and background seating).
- `outputs/`: Generated output images, comparative montages, feature arrays (`.npy`), bounding box records (`.json`), and measurement metrics (`.csv`).
- `Shaurya_Singh_CCA3_Submission.zip`: Submission ZIP containing the code, data, and derived outputs.

---

## 🔬 Computer Vision Tasks Implemented

### Task 1A: Noise Generation and Image Restoration
- **Noise Types:** Gaussian sensor noise ($\mu = 0, \sigma = 25$) and Impulse / Salt-and-Pepper noise (5% density).
- **Restoration Filters:** Median blur ($5\times 5$) and Gaussian blur ($5\times 5, \sigma = 1.2$).
- **Quantitative Metrics:** Peak Signal-to-Noise Ratio (PSNR dB) and Structural Similarity Index (SSIM).
- **Key Finding:** Median filtering strongly outperforms Gaussian filtering on salt-and-pepper noise ($\text{PSNR} \approx 45.12\text{ dB}, \text{SSIM} \approx 0.9911$), effectively eliminating isolated extreme pixel outliers without excessive blur. Gaussian filtering offers slight smoothing advantages on uniform Gaussian sensor noise ($\text{PSNR} \approx 32.00\text{ dB}$).

### Task 1B: Morphological Operations, Canny Edge Detection & Watershed Segmentation
- **Otsu Thresholding:** Automatically derives an optimal global threshold ($T = 101.0$) for binarization.
- **Morphology:** Evaluates Erosion, Dilation, Morphological Opening, and Morphological Closing using structured kernels.
- **Canny Edge Detector:** Multi-stage edge detection with hysteresis thresholding ($T_{\text{low}} = 100, T_{\text{high}} = 200$).
- **Distance Transform & Watershed:** Computes Euclidean distance transform, identifies regional maxima as seed markers, and performs topological watershed flooding to inspect segmentation boundaries and over-segmentation behavior.

### Task 2: Feature Extraction, Descriptors & Matching (HOG & SIFT)
- **Histogram of Oriented Gradients (HOG):** Extracted local gradient orientation histograms ($9$ orientations, $8\times 8$ cells, $2\times 2$ blocks), producing a feature vector of length $306,180$.
- **Scale-Invariant Feature Transform (SIFT):** Detected multi-scale keypoints and computed 128-dimensional local gradient descriptors ($235$ reference keypoints).
- **Feature Matching & Homography:** Uses Brute-Force Matcher with Lowe's ratio test ($d_1 < 0.75 d_2$) and RANSAC homography estimation across:
  1. Second original portrait: 7 matches, 5 RANSAC inliers (inlier ratio: 0.7143).
  2. $90^\circ$ clockwise rotation: 9 matches, 6 RANSAC inliers (inlier ratio: 0.6667).
  3. $70\%$ scale reduction: 6 matches, 4 RANSAC inliers (inlier ratio: 0.6667).

### Task 3: Object Recognition & Face Detection (Haar Cascade vs. YOLOv8)
- **Haar Frontal Face Cascade:** Hand-crafted rectangular Haar-like feature cascade localizes human faces. Successfully detected 1 frontal face in Photo 1 ($267.29\text{ ms}$) and 1 face in Photo 2 ($71.74\text{ ms}$).
- **YOLOv8n (Deep Learning):** Pretrained deep convolutional object detection model trained on COCO:
  - In Photo 1: Detected foreground person with high confidence ($\sim 0.94$).
  - In Photo 2: Detected foreground person ($\sim 0.88$) and successfully resolved 11 additional contextual background objects including banquet chairs, dining table, and attendees.

### Task 4: Video Motion Estimation (Architecture & Specification)
- Implemented Frame Differencing, Gaussian Mixture Model background subtraction (MOG2), and sparse Lucas-Kanade optical flow tracking.
- As per assignment requirements, video inputs remain pending until an original 15-30 second recording is captured and supplied. No fabricated motion data is recorded.

---

## 🚀 Installation & Execution

### 1. Environment Setup
Recommended Python version: `3.10` - `3.13`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Running the Interactive Notebook
Open `Shaurya_Singh_CCA3_Computer_Vision.ipynb` in VS Code or JupyterLab, select the `.venv` kernel, and execute cells from top to bottom. All outputs are pre-rendered and viewable directly in GitHub or VS Code.

### 3. Running the Standalone Pipeline
Run the complete automated pipeline via CLI:

```powershell
python pipeline.py --photo data/originals/photo1.png --second-photo data/originals/photo2.jpg --foreground light --output outputs/final_run
```

To include an original video and touching-objects photo when available:
```powershell
python pipeline.py --photo data/originals/photo1.png --second-photo data/originals/photo2.jpg --video data/originals/motion.mp4 --segmentation-photo data/originals/coins.jpg --foreground dark --output outputs/final_run
```

---

## 📜 References
- OpenCV Python Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- Scikit-Image Feature & Metric Documentation: https://scikit-image.org/
- Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/
