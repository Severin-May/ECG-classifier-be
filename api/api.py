from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import pandas as pd

from testers.predict import predict_and_evaluate

app_api = Flask(__name__)
CORS(app_api)

""" api endpoint handler """


@app_api.route('/api/v1/file-upload', methods=['POST'])
def upload_file():
    print("api enter")

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    model_type = request.form.get('model')
    print(model_type)

    if file:
        try:
            file_content = file.read()
            df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))

            results = predict_and_evaluate(model_type, df, True)
            return jsonify(results), 200
        except Exception as e:
            return jsonify({"error": "Error processing file: " + str(e)}), 400
    else:
        return jsonify({"error": "No file"}), 400


if __name__ == '__main__':
    app_api.run(debug=True)
