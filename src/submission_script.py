__author__ = "Robin Brecht, Philipp Paraguya"
__credits__ = ["Robin Brecht", "Philipp Paraguya"]

import json
import pandas as pd
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score,  make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import clone

from lightgbm import LGBMClassifier

pd.options.mode.chained_assignment = None  # default='warn'


def squared_cohen_kappa(y1, y2):
    """
    TODO Kommentieren
    """
    return cohen_kappa_score(y1,y2)**2


def import_data(pDATA_PATH):
    """
    TODO Kommentieren
    """
    print('Entered import_data')
    df = pd.read_csv(pDATA_PATH + '/train/train.csv')
    train_id = df['PetID']

    sentiment_mag = []
    sentiment_score = []
    for pet in train_id:
        try:
            with open(pDATA_PATH + '/train_sentiment/' + pet + '.json', 'r', encoding='UTF-8') as f:
                sentiment = json.load(f)
                # print(DATA_PATH+'/train_sentiment/' + pet + '.json')
            sentiment_mag.append(sentiment['documentSentiment']['magnitude'])
            sentiment_score.append(sentiment['documentSentiment']['score'])
        except FileNotFoundError:
            sentiment_mag.append(-1)
            sentiment_score.append(-1)

    df.loc[:, 'sentiment_mag'] = sentiment_mag
    df.loc[:, 'sentiment_score'] = sentiment_score

    df.head()

    vertex_xs = []
    vertex_ys = []
    bounding_confidences = []
    bounding_importance_fracs = []
    dominant_blues = []
    dominant_greens = []
    dominant_reds = []
    dominant_pixel_fracs = []
    dominant_scores = []
    label_descriptions = []
    label_scores = []

    for pet in train_id:
        try:
            with open(pDATA_PATH + '/train_metadata/' + pet + '-1.json', 'r', encoding='UTF-8') as f:
                data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                label_descriptions.append('nothing')
                label_scores.append(-1)
        except FileNotFoundError:
            vertex_xs.append(-1)
            vertex_ys.append(-1)
            bounding_confidences.append(-1)
            bounding_importance_fracs.append(-1)
            dominant_blues.append(-1)
            dominant_greens.append(-1)
            dominant_reds.append(-1)
            dominant_pixel_fracs.append(-1)
            dominant_scores.append(-1)
            label_descriptions.append('nothing')
            label_scores.append(-1)

    df.loc[:, 'vertex_x'] = vertex_xs
    df.loc[:, 'vertex_y'] = vertex_ys
    df.loc[:, 'bounding_confidence'] = bounding_confidences
    df.loc[:, 'bounding_importance'] = bounding_importance_fracs
    df.loc[:, 'dominant_blue'] = dominant_blues
    df.loc[:, 'dominant_green'] = dominant_greens
    df.loc[:, 'dominant_red'] = dominant_reds
    df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
    df.loc[:, 'dominant_score'] = dominant_scores
    df.loc[:, 'label_description'] = label_descriptions
    df.loc[:, 'label_score'] = label_scores

    return df


def feat_eng(df):
    print('Entered feat_eng')
    descriptions = df.Description.fillna("no_desc").values
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b', use_idf=True)
    X = vectorizer.fit_transform(list(descriptions))

    print('X:', X.shape)
    print(vectorizer.get_feature_names())

    svd = TruncatedSVD(n_components=300)
    svd.fit(X)
    X = svd.transform(X)

    print('X:', X.shape)

    X = pd.DataFrame(X, columns=['svd_{}'.format(i) for i in range(300)])
    df = pd.concat((df, X), axis=1)

    return df


def do_grid_search(lgbm, X, y):
    grid = {'learning_rate': [0.1, 0.001, 0.003, 0.0005], 'max_bin': [100, 255, 400, 500],
            'num_iterations': [50, 100, 150, 200, 300, 500]}

    kappa_scorer = make_scorer(squared_cohen_kappa)
    lgbm_cv = GridSearchCV(lgbm, grid, scoring=kappa_scorer, cv=3, verbose=10)
    lgbm_cv.fit(X, y)

    print("tuned hyperparameters :(best parameters) ", lgbm_cv.best_params_)
    print("accuracy :", lgbm_cv.best_score_)

    return lgbm_cv

def pred_ensemble(model_list, X):
    pred_full = 0
    for model in model_list:
        pred = model.predict(X)
        pred_full += pred

    return pred_full / len(model_list)

def train_and_run_cv(model, X, y, cv=3):
    print('Entered train_and_run_cv')
    skf = StratifiedKFold(n_splits=3)
    i = 0
    cv_score = []
    model_list = []

    for train_index, test_index in skf.split(X, y):
        i += 1
        print("training fold {} of {}".format(i, cv))
        X_train, X_test = np.array(X)[train_index, :], np.array(X)[test_index, :]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        model_copy = clone(model)
        model_copy.fit(X_train, y_train)
        score = squared_cohen_kappa(y_test, model_copy.predict(X_test))
        print("Score:", score)
        cv_score.append(score)
        model_list.append(model_copy)

    print("Mean cv Score", np.mean(cv_score))

    return model_list


def main(argv):
    """
    Diese Funktion ist der Einstiegspunkt für dieses Projekt
    """

    DATA_PATH = argv[0]

    df = import_data(DATA_PATH)
    df = feat_eng(df)

    df_train, df_test = train_test_split(df, test_size=0.2)

    drop_feat_list = ['AdoptionSpeed', 'Name', 'RescuerID', 'Description', 'PetID', 'label_description']

    feature_list = list(df.columns)
    feature_list = [x for x in feature_list if x not in drop_feat_list]

    X = df_train[feature_list]
    y = df_train['AdoptionSpeed'].values

    lgbm = LGBMClassifier(objective='multiclass', random_state=5)

    lgbm_list  = train_and_run_cv(lgbm, X, y)

    #lgbm = do_grid_search(X, y)

    #df_test['lgbm_pred'] = lgbm.predict(df_test[feature_list])
    df_test['lgbm_pred'] = pred_ensemble(lgbm_list, df_test[feature_list])

    lgbm_kappa = squared_cohen_kappa(df_test['AdoptionSpeed'], np.round(df_test['lgbm_pred'],0))
    print('Model tested! Squared Cohen Kappa: ' + str(lgbm_kappa))

    df_test[['PetID', 'lgbm_pred']].to_csv('submission.csv', index=False, header=['PetID', 'AdoptionSpeed'])

    #feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_, X.columns)), columns=['Value', 'Feature'])

    #print(feature_imp)




if __name__ == '__main__':
    main(sys.argv[1:])