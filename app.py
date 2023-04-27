from flask import Flask, request, jsonify
import pickle 
import numpy as np

from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()
my_file = THIS_FOLDER / "model.pkl"

model = pickle.load(open(my_file,'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"


@app.route('/predict', methods=["POST"])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')

    # result = {'cgpa':cgpa, 'iq':iq, 'profile_score':profile_score}
    input_query = np.array([[cgpa,iq,profile_score]])
    b = np.array(input_query, dtype=float)
    result = model.predict(b)[0]
    
    return jsonify({'placement':str(result)})

if __name__=='__main__':
    app.run(debug=True)