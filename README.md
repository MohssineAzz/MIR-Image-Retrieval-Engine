# MIR Project: Image Indexing and Retrieval Engine

## Overview

This project implements a Multimedia Information Retrieval (MIR) system capable of indexing and retrieving images based on visual content. It includes implementations of various feature extraction methods (Color Histograms, HOG, deep learning features) and a retrieval engine.

## Features

- **Image Indexing**: Extract features from image datasets.
- **Feature Extraction**:
  - Traditional Computer Vision descriptors (Histograms, HOG, etc.)
  - Deep Learning based extractors (ResNet, InceptionV4).
- **Search & Retrieval**: Query the database with an image to find similar matches.
- **GUI Interface**: User-friendly interface to perform searches.

## Installation

### Prerequisites
- Python 3.x
- Conda (optional but recommended)

### Setup

1.  **Clone the repository** (if you haven't already).
2.  **Install dependencies**:
    
    Using Conda:
    ```bash
    conda env create -n MIR --file requirements/requirements.yml
    conda activate MIR
    ```

    Or using pip:
    ```bash
    pip install -r requirements/requirements.txt
    ```

## Usage

### Running the Application

You can launch the application using the provided scripts depending on your operating system:

*   **macOS/Linux**:
    ```bash
    ./launch_LINUX_MACOS_with_conda.sh
    # or
    ./launch_LINUX_MACOS_without_conda.sh
    ```

*   **Windows**:
    Run `launch Windows - with conda.bat` or `launch Windows - without conda.bat`.

### Manual Execution

To run the interface directly from Python:

```bash
python src/interface.py
```

## Structure

- `src/`: Source code for the engine.
  - `feature_extractor.py`: Logic for extracting image features.
  - `retrieval.py`: Search and retrieval logic.
  - `interface.py`: Main GUI application.
  - `deep_learning/`: Notebooks and scripts for DL models.
- `requirements/`: Dependency files.
- `launch_*`: Scripts to easily start the project.

## Authors

Mohssine Azzizi/Mathieu Zilli



