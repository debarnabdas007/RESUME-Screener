from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline
import PyPDF2

app = Flask(__name__)

# Helper to extract text from PDF
def input_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page_text = reader.pages[page].extract_text()
        text += str(page_text)
    return text

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # 1. Get Input (either text paste or file upload)
        text_input = request.form.get("resume_text")
        file_input = request.files.get("resume_file")

        final_text = ""

        if file_input and file_input.filename != "":
            # Handle PDF
            final_text = input_pdf_text(file_input)
        elif text_input:
            # Handle Paste
            final_text = text_input
        else:
            return render_template('home.html', result="Please provide a resume!")

        # 2. Run Pipeline
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(final_text)

        # 3. Return Result
        return render_template('home.html', result=f"Predicted Profile: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)