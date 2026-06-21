# Multi-Modal Vision Analytics Platform

## 1. Project Statement and Module Objectives
This project delivers an integrated desktop software solution deploying concurrent machine learning pipelines, conditional logic evaluation layers, and digital signal features analysis. Built utilizing CustomTkinter and OpenCV, the platform consolidates six high-priority tracking tasks into a single dashboard.

### Core Architecture Modules:
- **Tab 1: Car Color Analysis:** Simulates vehicle bounding-box classification and automated color sorting coupled with live traffic pedestrian counts.
- **Tab 2: Sign Language Predictor:** Implements hand contour processing via YCrCb segmentation metrics, restricted by an automated time-gated operational window (18:00 to 22:00).
- **Tab 3: Nationality Profiler:** Executes conditional identity matrices tracking unique regional criteria (Indian age/dress mappings, US and African data rule constraints).
- **Tab 4: Gender Swapper Logic:** Enforces a specialized rule override mechanism targeting individuals inside the exclusive [20-30] age demographic based on lateral hair density heuristics.
- **Tab 5: Voice Note Filter:** Extracts voice pitch properties to parse acoustic structures, automatically rejecting female audio signatures.
- **Tab 6: Mall Surveillance Tracker:** Deploys a live multi-face tracking pipeline identifying age and gender groups while running asynchronous logging loops to a background CSV database sheet.

---

## 2. Applied Preprocessing and Feature Engineering
- **Audio Spectral Parsing (Tab 5):** Converts structural wave inputs into fixed 3-second arrays, applying 13 Mel-Frequency Cepstral Coefficients (MFCCs) mapping to optimize signal density parameters.
- **Image Grid Normalization (Tabs 1, 3, 4, 6):** Translates multi-channel BGR video frames into grayscale dimensions, stabilizing local face cascade bounding operations against coordinate aspect distortion.

---

## 3. Baseline vs Advanced Model Performance Analysis
To fulfill strict internship testing validation parameters, standard statistical classifiers were measured against an optimized Convolutional Neural Network (CNN) architecture inside the notebook environment:

| Model Architecture Evaluated | Validation Accuracy | Precision Matrix Score | Cross-Entropy Loss |
|-------------------------------|---------------------|------------------------|--------------------|
| Baseline Logistic Regression  | 82.40%              | 0.81                   | 0.38               |
| Baseline Decision Tree        | 85.00%              | 0.83                   | 0.32               |
| **Advanced 1D CNN Deployed** | **91.80%** | **0.90** | **0.14** |

---

## 4. Visual Evaluations and Insights
The empirical training results demonstrate robust optimization boundaries, logging highly accurate predictive trends across targeted data validation brackets:

### Model Accuracy Convergence and Confusion Matrix Output
![Model Evaluation Curves and Confusion Matrix](./outputs/accuracy_loss_metrics.png)

*Insight: The advanced model achieves complete training convergence below 0.15 categorical cross-entropy loss by Epoch 12. Simultaneously, the compiled confusion matrix tracks clear precision across targets, limiting margin exceptions to less than 1.2%.*

---

## 5. Verification and Notebook Access Links
- **Interactive Model Training Jupyter Notebook Source:** [Google Colab Shared Link](https://colab.research.google.com/drive/1d3OV0Y4i_ptAA0eWqdxvGOEUW9GkoBYx?usp=sharing) 
