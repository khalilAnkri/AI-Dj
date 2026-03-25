import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Spotify Hit Predictor")

model = pickle.load(open("../training/model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))
list_top5 = pickle.load(open("top_5.pkl", "rb"))

classes = ["Hit", "Not a Hit"]
history = []

@app.route("/")
def home():
    return{"Homepage of Spotify Hit Predictor"}
#   return render_template("homepage.html")

@app.get("/features")
def get_top_features():
    return {"top_5": list_top5}


@app.post("/predict")
def predict(data: dict):
    df_input = pd.DataFrame([data], columns=model_columns)

    prediction = int(model.predict(df_input)[0]) # index
    name = classes[prediction]

    result = {
            "id": len(history),
            "input": data,
            "class_index": prediction,
            "class_name": name
        }

    history.append(result)
    return result

@app.get("/past_predictions/{prediction_id}")
def get_prediction_by_id(prediction_id: int):

    if prediction_id < 0 or prediction_id >= len(history):
        raise HTTPException(status_code=404, detail="Prediction ID not found")

    return history[prediction_id]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
