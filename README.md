# **EF-VLNet: Multi-scale Vision-Language Fusion for Cardiac Function Assessment**

1. **Proposed a novel multi-scale semantic fusion framework**: Utilizing a hierarchical feature learning mechanism, the framework simultaneously captures macro-dynamic features and micro-motion patterns of cardiac objects. An attention-based fusion module establishes complex semantic relationships between coarse-grained and fine-grained features.
2. **Systematic integration of Vision-Language Models (VLM)**: For the first time, VLMs are integrated into echocardiogram analysis. Through medically-guided feature embeddings, the system achieves deep semantic alignment between visual features and clinical terminology, significantly improving model interpretability and clinical relevance.
3. **End-to-end trainable architecture**: Designed to map raw ultrasound videos directly to Ejection Fraction (EF) predictions while maintaining physiological plausibility. Extensive experiments on public datasets demonstrate superior performance over state-of-the-art methods, particularly in complex clinical cases.

## Core Features

1. **Automated Report Generation**: Utilizes `Lingshu-7B` (a medical large multi-modal model) to convert structured cardiac measurement data into natural medical ultrasound descriptions and diagnostic conclusions.
2. **First-frame Annotation**: Employs `LabelMe` software to mark the initial frames of the `EchoNet-Dynamic` chamber videos.
3. **Automated Four-Chamber Segmentation**: Uses `MedSAM2` (a medical segmentation foundation model) for fast, automated segmentation of `EchoNet-Dynamic` videos to create a fully labeled dataset.
4. **Multi-modal Feature Fusion**: Integrates four distinct data streams using `MedSAM2`:
   - **Visual Stream**: Pre-extracted R2Plus1D or 3D ResNet (R3D) video features.
   - **Semantic Stream**: Vectorization of AI-generated diagnostic reports using BioBERT.
   - **Temporal Stream**: Geometric dynamic data including area changes, flow velocity, and acceleration of each cardiac chamber.
   - **Dynamic Stream**: Time-point information based on the cardiac cycle (Systole/Diastole).

------

## Technical Architecture

The system aggregates various dimensions of features through the `ImprovedFourLayerFusionModel`:

- **Semantic Stream Encoder**: BioBERT (768-dim) -> Multi-Layer Perceptron (MLP).
- **Dynamic Stream Encoder**: Bi-LSTM + Multi-head Attention (independent modeling for each chamber).
- **Visual Stream Encoder**: Pre-trained R2Plus1D/R3D feature mapping.
- **Temporal Stream Encoder**: MLP.
- **Fusion Layer**: Multi-head Self-Attention (Attention Fusion) processing the concatenated global features.
- **Regression Layer**: Fully connected network outputting the predicted EF value (supports Sigmoid or Tanh activation).

------

## File Descriptions

### 1. Model and Report Generation

- **`download_model.py`**: Downloads and verifies the `Lingshu-7B` medical MLLM from HuggingFace. Features system resource checks and resume-download capabilities.
- **`ai_report.py`**: Core logic for report generation. It parses ultrasound measurements, constructs prompts, and calls the large model to generate professional ultrasound descriptions. Includes a rule-based fallback mechanism if the model is unavailable.

### 2. Feature Extraction

- **`biobert_extractor.py`**: Feature extractor based on `dmis-lab/biobert-v1.1`. Converts text reports into machine-understandable feature vectors, supporting Mean/CLS/Max pooling and feature caching.

### 3. EF Prediction Models

The system provides several versions of prediction scripts to accommodate different model structures and activation functions:

- **`EF_Prediction_MultiModal_R2Plus1d_Fusion_Sigmoid.py`**: Uses R2Plus1D visual features with a Sigmoid output layer (normalized output).
- **`EF_Prediction_MultiModal_R2Plus1d_Fusion_Tanh.py`**: Uses Tanh activation with a Sigmoid output layer.
- **`EF_Prediction_MultiModal_R3D_Fusion_Sigmoid.py`**: Version utilizing 3D ResNet (R3D) video features.

------

## Installation and Preparation

### Environment Requirements

- Python 3.8+
- PyTorch (CUDA support recommended)
- Transformers (HuggingFace)
- Scikit-learn, Pandas, NumPy
- Matplotlib (for result visualization)

### Model Download

Run the following command to download the Lingshu-7B model:

Bash

```
python download_model.py
```

------

## Usage Guide

### Step 1: Generate Diagnostic Reports

First, convert video analysis results into text descriptions to allow for semantic feature extraction:

Bash

```
python ai_report.py
```

### Step 2: Train the EF Prediction Model

Using the R2Plus1D version as an example, start the multi-modal fusion training:

Bash

```
python EF_Prediction_MultiModal_R2Plus1d_Fusion_Sigmoid.py \
    --base_dir /path/to/EchoNet-Dynamic \
    --batch_size 32 \
    --epochs 200 \
    --save_dir ./checkpoints
```

### Step 3: Testing and Visualization

After training, the script automatically generates a test set report (`test_results.json`) and renders:

- **Scatter Plot**: Predicted EF vs. Actual EF.
- **ROC Curve**: Classification performance for different EF thresholds (e.g., 35%, 40%······).

------

## Data Requirements

The project is designed for the `EchoNet-Dynamic` dataset. The `base_dir` must contain:

- `FileList.csv`: Video filenames, split assignments, and EF labels.
- `t0_t1.csv` & `t1_t2.csv`: Cardiac cycle phase timing.
- `heart_chamber_flow_analysis_part*.csv`: Chamber area and blood flow analysis data.
- `features/`: Pre-extracted video feature files (.npz format).

------

## Dataset

This project uses the `EchoNet-Dynamic` dataset. You can download it from the official repository: [EchoNet-Dynamic GitHub](https://github.com/echonet/dynamic). Place the dataset inside the `data` folder. The data folder is accessed from the web address of the web disk: files shared through the web disk: data.zip
Link: https://pan.baidu.com/s/1zaA5LEkOFm3tjMDKXQAv7w?pwd=momo Extraction code: momo

Pre-trained models and key data for this project are included in the `checkpoints_r2plus1d_sigmoid` folder  and  `checkpoints_r2plus1d_tanh` folder   for immediate use, or you can train and adjust them yourself.


##  Demo
<video width="800" controls>
  <source src="2f96c36aae7cd621fff9acad5d5bdaca.mp4" type="video/mp4">
</video>


## Important Notes

- **Path Configuration**: Before use, please search for `/path/to/your/own/...` placeholders in the code and modify them to your local absolute paths.

------

## References and Acknowledgments

```

```
