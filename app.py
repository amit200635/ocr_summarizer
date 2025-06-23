from flask import Flask, request, render_template, url_for
import os
from ocr_summarizer import process_file  # ✅ Use your actual file name

app = Flask(__name__)

# Create folders if they don't exist
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    summary = None
    keywords = []
    docx_link = None

    if request.method == 'POST':
        file = request.files['pdf_file']
        if file:
            ext = file.filename.split('.')[-1].lower()
            allowed = {'pdf', 'docx', 'jpg', 'jpeg', 'png', 'txt'}

            if ext in allowed:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Process the uploaded file using your OCR + summarizer
                output_path, summary, keyword_str = process_file(file_path)
                keywords = keyword_str.split(', ')
                docx_link = os.path.basename(output_path)

    return render_template('index.html', summary=summary, keywords=keywords, docx_link=docx_link)


if __name__ == '__main__':
    app.run(debug=True)
