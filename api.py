from flask import Flask, request
from sklearn.externals import joblib
import json
app = Flask(__name__)

# Load models
print "Loading models from file..."
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
# Attach app endpoint
@app.route("/classify")
def classify():
    if 'q' in request.args:
        data = request.args['q']
        x = vectorizer.transform([data])
        output = model.predict_proba(x)[0]
        result = {
            "not_name": output[0],
            "name": output[1]
        }
        print result
        return json.dumps(result)
    else:
        return "Please add a query `q` to classify"

if __name__ == "__main__":
    app.run()
