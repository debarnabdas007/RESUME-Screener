# 🚀 AI Resume Screener & Job Matcher

A robust Machine Learning application that classifies resumes into professional categories (e.g., Data Science, Java Developer) and calculates a **Match Score** against a specific Job Description (JD) using TF-IDF vectorization and Cosine Similarity.

## 🌟 Features

* **Resume Classification:** Automatically categorizes resumes into 20+ domains (Data Science, HR, Web Dev, etc.) with 98% accuracy.
* **Job Match Score:** Compares the Resume against a provided Job Description to calculate a relevance percentage.
* **PDF Parsing:** Extracts text seamlessly from uploaded PDF resumes.
* **Production Pipeline:** Modular code structure (Ingestion -> Transformation -> Training -> Prediction).
* **Modern UI:** Flask Web App with **Dark Mode** support and responsive design.

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Web Framework:** Flask
* **Machine Learning:** Scikit-Learn (KNN, TF-IDF, OneVsRest)
* **Data Processing:** Pandas, NumPy, Regex
* **PDF Handling:** PyPDF2

## 📂 Project Structure

```bash
Resume_Screener/
├── artifacts/          # Stores trained models (.pkl files)
├── notebook/           # Jupyter notebooks for EDA and Sandboxing
├── src/                # Source code
│   ├── components/     # Data Ingestion, Transformation, Trainer
│   ├── pipeline/       # Prediction Pipeline logic
│   ├── utils.py        # Helper functions (save/load objects)
├── templates/          # HTML files (Dark Mode enabled)
├── app.py              # Flask Application entry point
├── requirements.txt    # Project dependencies
└── README.md           # Documentation



# 🚀 How to Run Locally
1. Clone the Repo:

git clone [https://github.com/your-username/resume-screener.git](https://github.com/your-username/resume-screener.git)
cd resume-screener

2. Create Virtual Environment:

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

3. Install Dependencies:

pip install -r requirements.txt

4. Train the Model (Optional)
If you want to retrain the model from scratch:

# Run the pipeline to generate artifacts
python -m src.components.model_trainer

5. Run the App
Bash
python app.py



📊 Understanding the Match Score
The app uses TF-IDF Vectorization (3000 features) and Cosine Similarity.

> 30%: Excellent Match (High keyword overlap in critical technical terms).

15% - 30%: Good Match (Relevant context found).

< 15%: Low Match (Vocabulary gap or domain mismatch).

👨‍💻 Author
Debarnab Das