from flask import Flask, request
from sklearn.externals import joblib
import json
app = Flask(__name__)
# Load models
models = {
    "multi_nb": None,
    "ridge_classifier": None,
    "knn": None,
    "random_forest": None
}
print "Loading models from file..."
for model in models:
    print "Loading ", model
    models[model] = joblib.load('model/' + model + '.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
# Attach app endpoint
@app.route("/classify")
def classify():
    if 'q' in request.args:
        query = request.args['q']
        x = vectorizer.transform([query])
        outputs = {}
        print models
        for model in models:
            try:
                outputs[model] = models[model].predict_proba(x)[0].tolist()
            except:
                outputs[model] = models[model].predict(x)[0].tolist()
        return json.dumps(outputs)
    else:
        return "Please add a query `q` to classify"

if __name__ == "__main__":
    app.run()
