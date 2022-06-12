from sklearn.preprocessing import StandardScaler
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib
from imblearn.pipeline import Pipeline as imbpipeline
import pickle
from constants import MODEL_PATH


def train_classifier(X, y, model_path=MODEL_PATH):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=11)
    if not os.path.exists(model_path):
        clf = ExtraTreesClassifier()



        pipeline = imbpipeline(steps=[['smote', SMOTE(random_state=11)],
                                      ['classifier', clf]])

        param_grid = dict(classifier__max_depth=range(6, 14, 2),
                          classifier__n_estimators=[1000])
        print(54*'#')
        print('Cross validation')

        grid_search = GridSearchCV(estimator=pipeline,
                                   param_grid=param_grid,
                                   scoring='roc_auc',
                                   cv=3,
                                   n_jobs=10,
                                   verbose=5)

        grid_search.fit(X_train, y_train)
        cv_score = grid_search.best_score_
        test_score = grid_search.score(X_test, y_test)
        print(54 * '#')
        clf = grid_search.best_params_
        print(54 * '#')
        classifier_params = [(key,value) for key,value in clf.items()]
        print(classifier_params)
        final_classifier = ExtraTreesClassifier(max_depth=classifier_params[0][1], n_estimators=classifier_params[1][1])
        final_classifier.fit(X_train, y_train)
        print(54 * '#')
        pickle.dump(final_classifier, open(model_path, 'wb'))

        print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')

    else:
        clf = joblib.load(open(MODEL_PATH, 'rb'))
        #final_classifier = ExtraTreesClassifier(max_depth=4, n_estimators=50)
        #final_classifier.fit(X_train, y_train)
    return clf, X_test, y_test