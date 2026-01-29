# 🚀 AI Resume Screener & Job Matcher

An intelligent Machine Learning application that automatically **classifies resumes** into professional categories (Data Science, Java Developer, HR, etc.) and calculates a **Match Score** against specific Job Descriptions using TF-IDF vectorization and Cosine Similarity.

---
## Live Demo (Preview)

> If the live link is down, see screenshots below 👇

![Home Page](images\Home Page.png)
![Prediction Result](images\Prediction.png)



## 🌟 Key Features

✅ **Resume Classification**
- Automatically categorizes resumes into 20+ professional domains
- Achieves 98% accuracy using KNN with One-vs-Rest classifier
- Supports both PDF uploads and text input

✅ **Job Match Scoring**
- Compares resume against Job Description using Cosine Similarity
- Calculates relevance percentage (0-100%)
- Helps identify best-fit candidates for specific roles

✅ **PDF Processing**
- Seamless extraction of text from uploaded PDF files
- Robust text cleaning and preprocessing pipeline
- Handles multi-page PDFs

✅ **Production-Ready Architecture**
- Modular code structure following ML pipeline best practices
- Separation of concerns: Ingestion → Transformation → Training → Prediction
- Comprehensive error handling and logging

✅ **Modern Web Interface**
- Flask-based web application
- Dark Mode support with responsive design
- User-friendly dashboard for resume analysis
- Contact page for inquiries

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Web Framework** | Flask |
| **ML Models** | Scikit-Learn (KNN, TF-IDF Vectorizer, OneVsRest) |
| **Data Processing** | Pandas, NumPy, Regex |
| **PDF Handling** | PyPDF2 |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Package Manager** | pip + setuptools |

---

## 📂 Project Structure

```
RESUME-Screener/
│
├── 📦 artifacts/              # Generated files & trained models
│   ├── data.csv              # Raw dataset (reference)
│   ├── train.csv             # Training dataset (80%)
│   ├── test.csv              # Test dataset (20%)
│   ├── model.pkl             # Trained KNN classifier
│   ├── preprocessor.pkl      # TF-IDF vectorizer
│   └── label_encoder.pkl     # Category encoder
│
├── 📓 notebooks/              # Jupyter notebooks for experimentation
│   ├── 1_EDA.ipynb           # Exploratory Data Analysis
│   ├── 2_Model_Train.ipynb   # Model prototyping & training
│   └── data/                 # Raw datasets for notebook use
│
├── 🔧 src/                    # Production source code
│   ├── components/            # Core ML logic
│   │   ├── data_ingestion.py      # Load & split data (80-20)
│   │   ├── data_transformation.py # Text cleaning & TF-IDF vectorization
│   │   └── model_trainer.py       # Train KNN & save artifacts
│   │
│   ├── pipeline/              # Orchestration layer
│   │   ├── train_pipeline.py      # Execute: Ingest → Transform → Train
│   │   └── predict_pipeline.py    # Execute: Clean → Vectorize → Predict + Score
│   │
│   ├── logger.py              # Centralized logging configuration
│   ├── exception.py           # Custom exception handling
│   ├── utils.py               # Helper functions (save/load objects)
│   └── __init__.py            # Package initialization
│
├── 🎨 templates/              # HTML templates
│   ├── home.html              # Main dashboard with dark mode
│   └── contact.html           # Contact page
│
├── 📋 app.py                  # Flask application entry point
├── 🔐 setup.py                # Package installation configuration
├── 📝 requirements.txt         # Project dependencies
├── 📚 ARCHITECTURE.md          # Detailed architecture documentation
├── 🔍 README.md               # This file (original)
└── 📋 new_README.md           # Enhanced documentation
```

---

## ⚙️ How It Works

### 1️⃣ **Resume Classification Pipeline**

```
User Input (PDF/Text)
    ↓
[Extract Text] (PyPDF2)
    ↓
[Clean Text] (Regex, Stop words removal)
    ↓
[Vectorize] (TF-IDF: 3000 features)
    ↓
[KNN Classification] (OneVsRest Classifier)
    ↓
Category Output (e.g., "Data Science", "Java Developer")
```

### 2️⃣ **Job Match Scoring Pipeline**

```
Resume + Job Description
    ↓
[Clean Both Texts]
    ↓
[Vectorize Both] (Same TF-IDF model)
    ↓
[Calculate Cosine Similarity]
    ↓
Match Score (0-100%)
```

### 3️⃣ **Understanding Match Scores**

| Score Range | Interpretation |
|------------|-----------------|
| **> 30%** | 🟢 **Excellent Match** - High keyword overlap in critical technical terms |
| **15% - 30%** | 🟡 **Good Match** - Relevant context and experience found |
| **< 15%** | 🔴 **Low Match** - Vocabulary gap or domain mismatch |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/debarnabdas007/RESUME-Screener.git
cd RESUME-Screener
```

### Step 2: Create Virtual Environment (Recommended)

**On Windows:**
```powershell
python -m venv resume_venv
resume_venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv resume_venv
source resume_venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including Flask, Scikit-Learn, Pandas, NumPy, and PyPDF2.

### Step 4: Train the Model (First Time Only)

If you want to retrain the model from scratch with your own dataset:

```bash
python -m src.pipeline.train_pipeline
```

This will:
- Load data from `notebooks/data/UpdatedResumeDataSet.csv`
- Split into 80% training and 20% testing
- Train the KNN classifier
- Save artifacts to `artifacts/` folder

### Step 5: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

Open your browser and navigate to the URL to access the dashboard.

---

## 📖 Usage Guide

### Via Web Interface

1. **Navigate to Home Page** (`http://localhost:5000`)
2. **Upload Resume**
   - Option A: Upload a PDF file
   - Option B: Paste resume text directly
3. **Enter Job Description** (Optional)
   - Paste the target job description to get a match score
4. **Submit**
   - View classification and match score

### Via Python Code

```python
from src.pipeline.predict_pipeline import PredictPipeline

# Initialize pipeline
pipeline = PredictPipeline()

# Predict with optional job description
resume_text = "Your resume text here..."
job_description = "Your JD here..."

category, match_score = pipeline.predict(resume_text, job_description)

print(f"Category: {category}")
print(f"Match Score: {match_score}%")
```

---

## 🔍 Core Components

### `data_ingestion.py`
**Purpose:** Load and prepare dataset

**Key Functions:**
- `initiate_data_ingestion()` - Reads CSV, splits 80-20, saves artifacts

### `data_transformation.py`
**Purpose:** Clean text and create vectorizer

**Key Functions:**
- `clean_resume_text()` - Remove special chars, convert to lowercase
- `initiate_data_transformation()` - Apply TF-IDF vectorization (3000 features)

### `model_trainer.py`
**Purpose:** Train and evaluate the classifier

**Key Functions:**
- `initiate_model_trainer()` - Train OneVsRest KNN, evaluate accuracy, save model

### `predict_pipeline.py`
**Purpose:** Make predictions on new resumes

**Key Functions:**
- `predict(resume_text, job_description)` - Classify resume and calculate match score

---

## 📊 Model Specifications

| Aspect | Details |
|--------|---------|
| **Algorithm** | K-Nearest Neighbors (KNN) |
| **Multi-class Strategy** | One-vs-Rest Classifier |
| **Feature Extraction** | TF-IDF Vectorizer (3000 features) |
| **Training Accuracy** | ~98% |
| **Similarity Metric** | Cosine Similarity |
| **Categories** | 20+ professional domains |

---

## 📝 Key Files Explained

### `app.py`
Flask application with two main routes:
- `GET /` - Display home page
- `POST /` - Process resume upload/text input
- `GET /contact` - Display contact page

### `requirements.txt`
Lists all Python dependencies. Install with:
```bash
pip install -r requirements.txt
```

### `setup.py`
Configures the package for local installation:
```bash
pip install -e .
```
This makes `src` importable as a package throughout the project.

### `ARCHITECTURE.md`
Detailed documentation of the project structure and design patterns.

---

## 🐛 Troubleshooting

**Issue:** Model files not found
```
Solution: Run the training pipeline first:
python -m src.pipeline.train_pipeline
```

**Issue:** PDF upload not working
```
Solution: Ensure PyPDF2 is installed:
pip install PyPDF2
```

**Issue:** "No module named 'src'"
```
Solution: Install the package in editable mode:
pip install -e .
```

**Issue:** Port 5000 already in use
```
Solution: Modify app.py to use a different port:
app.run(debug=True, port=5001)
```

---

## 📈 Performance Metrics

- **Resume Classification Accuracy:** 98%
- **Training Data Size:** 80% of dataset
- **Test Data Size:** 20% of dataset
- **Features (TF-IDF):** 3000
- **Algorithm:** OneVsRest KNN
- **Inference Time:** < 1 second per resume

---

## 🔐 Project Features

✨ **Robustness**
- Comprehensive error handling with custom exceptions
- Detailed logging at each step
- Graceful fallback when job description is not provided

✨ **Scalability**
- Modular architecture allows easy feature additions
- Pipeline design enables model retraining
- Vectorizer can be updated with new data

✨ **User Experience**
- Dark mode support for comfortable viewing
- Responsive design for mobile and desktop
- Clear, actionable results with match percentages

---

## 👨‍💻 Author & Contact

**Developer:** Debarnab Das

**GitHub:** [debarnabdas007](https://github.com/debarnabdas007)

**Repository:** [RESUME-Screener](https://github.com/debarnabdas007/RESUME-Screener)

For questions or suggestions, visit the contact page in the application.

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: Updated Resume Dataset (Kaggle)
- ML Framework: Scikit-Learn
- Web Framework: Flask
- PDF Processing: PyPDF2

---

## 📚 Additional Resources

- **Scikit-Learn Docs:** https://scikit-learn.org/
- **Flask Documentation:** https://flask.palletsprojects.com/
- **TF-IDF Vectorization:** https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- **Cosine Similarity:** https://scikit-learn.org/stable/modules/metrics.pairwise.html#cosine-similarity

---

**Last Updated:** January 29, 2026

**Version:** 0.0.1
