from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline
import PyPDF2

app = Flask(__name__)

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
        # 1. Get Inputs
        text_input = request.form.get("resume_text")
        file_input = request.files.get("resume_file")
        jd_input = request.form.get("jd_text") # <--- Get Job Description

        final_text = ""

        if file_input and file_input.filename != "":
            final_text = input_pdf_text(file_input)
        elif text_input:
            final_text = text_input
        else:
            return render_template('home.html', result="Please provide a resume!")

        # 2. Run Pipeline
        predict_pipeline = PredictPipeline()
        
        # Unpack the two return values
        predicted_category, match_score = predict_pipeline.predict(final_text, jd_input)

        # 3. Format Output
        result_string = f"Category: {predicted_category}"
        
        if match_score != "N/A":
            result_string += f" | Match Score: {match_score}%"

        return render_template('home.html', result=result_string)
    
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)