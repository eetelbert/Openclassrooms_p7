from model import train_classifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from constants import EXPLAINER_PATH, DASHBOARD_PATH







def create_explainer(X, y):
    clf, X_valid, y_valid = train_classifier(X, y)
    cate_cols = X.select_dtypes("category").columns.to_list()
    explainer = ClassifierExplainer(clf, X_valid, y_valid)
                        # shap='linear',  # manually set shap type, overrides default 'guess'
                        # X_background=X_train,  # set background dataset for shap calculations
                        # model_output='logodds',  # set model_output to logodds (vs probability)
                        # cats=['Sex', 'Deck', 'Embarked'],  # makes it easy to group onehotencoded vars
                        # idxs=test_names,  # index with str identifier
                        # index_name="Passenger",  # description of index
                        # descriptions=feature_descriptions,  # show long feature descriptions in hovers
                        # target='target',  # the name of the target variable (y)
                        # precision='float32',  # save memory by setting lower precision. Default is 'float64'
                        # labels=['Not survived', 'Survived'])
    return explainer


def create_dashboard(explainer, app):
    db = ExplainerDashboard(explainer, server=app, url_base_pathname="/dashboard/",
                            importances=False,
                            model_summary=False,
                            contributions=True,
                            whatif=False,
                            shap_dependence=False,
                            shap_interaction=False,
                            decision_trees=False)

    db.to_yaml(DASHBOARD_PATH, explainerfile=EXPLAINER_PATH, dump_explainer=True)
    return db




#db = ExplainerDashboard.from_config(DASHBOARD_PATH)
#app = db.flask_server()