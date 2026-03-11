from flask import Flask, request, jsonify, render_template
import pickle, numpy as np

app = Flask(__name__, template_folder="templates")

model = pickle.load(open("model.pkl", "rb"))
classes = ["setosa", "versicolor", "virginica"]
history = []

def predict_class(features):
    i = int(model.predict(np.array([features]))[0])
    return i, classes[i]

@app.route("/")
def home():
    return render_template("hello.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or len(data.get("features", [])) != 4:
        return jsonify({"error": "Provide 4 features"}), 400

    i, name = predict_class(data["features"])
    result = {"id": len(history), "input": data["features"], "class_index": i, "class_name": name}
    history.append(result)
    return jsonify(result)

@app.route("/past_predictions")
def past():
    return jsonify(history)

@app.route("/past_predictions/<int:id>", methods=["PUT"])
def update(id):
    if id >= len(history):
        return jsonify({"error": "Not found"}), 404

    data = request.get_json()
    if not data or len(data.get("features", [])) != 4:
        return jsonify({"error": "Provide 4 features"}), 400

    i, name = predict_class(data["features"])
    history[id] = {"id": id, "input": data["features"], "class_index": i, "class_name": name}
    return jsonify(history[id])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)