"""API server example

Usage from command line:

```bash
$ curl http://127.0.0.1:5001 -X POST -H "Content-Type: application/json"  -d '{"bathrooms": "2", "bedrooms": "3", "sqft_living": "1800",  "sqft_lot": "2200", "floors": "1", "waterfront": "1", "condition": "3"}'
1605832.1056204173%  

WARNING: CMD prompt does not support quotes, use \" instead
"""

import json
import logging
import os.path
import pickle

import pandas as pd
from flask import Flask, request

# -----------------------------------------------------------------------------
# API Server
# -----------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "api-server-secret-key"

# Model features used for prediction
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]


@app.route("/", methods=["POST"])
def index():
    """API function"""

    # model input
    args = request.json
    filt_args = {key: [int(args[key])] for key in FEATURES}
    df = pd.DataFrame.from_dict(filt_args)
    logging.info("User values: %s", filt_args)

    # prediction
    path = 'homework/house_predictor.pkl'
    with open(path, "rb") as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict(df)

    # result
    return str(prediction[0][0])


if __name__ == "__main__":
    app.run(debug=True, port=5001)