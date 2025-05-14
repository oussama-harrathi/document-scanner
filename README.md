# 📄 Document Scanner & Perspective Correction

This project is a Python-based document scanner that detects, extracts, and enhances a document from an image using OpenCV. It supports both automatic and manual modes for detecting document corners, applies perspective transformation, and performs contrast enhancement and binarization.

## 🔍 Features

- Automatic document boundary detection using contours  
- Interactive corner selection (fallback mode)  
- Perspective correction (top-down view)  
- CLAHE contrast enhancement and Otsu binarization  
- Clean black text on white background  
- Support for debugging and step-by-step visualization  
- CLI-based interface for different use cases

---


## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/oussama-harrathi/document-scanner
cd document-scanner
```

### 2. Create and Activate a Virtual Environment

Windows (PowerShell):

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies


```bash
pip install opencv-python numpy
```

---

## ▶️ Running the Application

### Automatic Mode (with contour detection)

```bash
python app.py path_to_image.jpg
```

### Interactive Mode (manual corner selection)

```bash
python app.py path_to_image.jpg --interactive
```

### Additional Flags

- `--no-enhance`: Skip post-warp contrast enhancement  
- `--show-steps`: Display processing steps visually  
- `--no-auto-preprocess`: Disable preprocessing (if applicable)

---

## 📂 Output

- Enhanced scans are saved in the `scanned_documents/` folder  
- Saved files include:  
  - `scanned_YYYYMMDD_HHMMSS.jpg` – final result  
  - `warped_orig_YYYYMMDD_HHMMSS.jpg` – warped (unprocessed) version

---


## 📚 Technologies Used

- Python 3.x  
- OpenCV  
- NumPy

