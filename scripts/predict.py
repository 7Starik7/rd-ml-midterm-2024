import pickle

from argparse import ArgumentParser
from flask import Flask, request, jsonify

app = Flask("Loan approval service")

input_file_path = 'dv_model.pkl'

with open(input_file_path, 'rb') as f:
    dict_vectorizer, model = pickle.load(f)
    print("Extracting model...")
    print(dict_vectorizer)
    print(model)


@app.route('/predict', methods=['POST'])
def predict_logistic_regression_model():
    loan_application = request.get_json()
    X = dict_vectorizer.transform(loan_application)
    y_prediction = model.predict_proba(X)[0, 1]
    y_prediction = y_prediction.round(2)

    if y_prediction >= 0.75:
        decision = "Approved"
    elif y_prediction <= 0.74:
        decision = "Rejected"
    else:
        return 'Unknown'

    result = {
        "decision": str(decision),
        "probability_rate": float(y_prediction)
    }
    return jsonify(result)


if __name__ == '__main__':
    parser = ArgumentParser(description="Make a prediction using a trained model")
    parser.add_argument('--path',
                        help=f"Specify path to dataset. '{input_file_path}' will be used by default.",
                        default=input_file_path)
    args = parser.parse_args()
    input_file_path = args.path
    app.run(debug=False, host='0.0.0.0', port=8024)
