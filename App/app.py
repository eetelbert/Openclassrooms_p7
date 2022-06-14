from constants import TEST_PATH, TRAIN_PATH
from dashboard import create_dashboard, create_explainer
from preprocessing import feature_engineering
from flask import Flask
import pandas as pd

app = Flask(__name__)

@app.route('/dashboard')
def return_dashboard(dashboard):
    return dashboard.app.index()

@app.route("/")
@app.route("/main")
def main():
    return "Hello Wolrd"
   # train_df = pd.read_csv(TRAIN_PATH)
   # test_df = pd.read_csv(TEST_PATH)
   # X, y = feature_engineering(train_df, test_df)
   # explainer = create_explainer(X, y)
   # db = create_dashboard(explainer, app)
   # return_dashboard(db)


if __name__ == "__main__":
    app.run()

# Docs :
# https://explainerdashboard.readthedocs.io/en/latest/explainers.html#parameters
# https://explainerdashboard.readthedocs.io/en/latest/custom.html#switching-off-tabs
# custom metrics :
# https://explainerdashboard.readthedocs.io/en/latest/custom.html#using-custom-metrics

# Exemple de dashboard dans heroku :
# http://titanicexplainer.herokuapp.com/

# Lire ça et suivre les instructions pour deployer mon explainer sur heroku :
# https://github.com/oegedijk/explainerdashboard/blob/479746e7caa5ebe5521b81536f65c41c02750acc/docs/source/deployment.rst
