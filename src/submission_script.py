__author__ = "Robin Brecht, Philipp Paraguya"
__credits__ = ["Robin Brecht", "Philipp Paraguya"]

import json
import pandas as pd
import sys
import numpy as np
import scipy as sp

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import clone

from functools import partial

from lightgbm import LGBMClassifier

pd.options.mode.chained_assignment = None  # default='warn'


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(y, y_pred):
    """
       Calculates the quadratic weighted kappa
       axquadratic_weighted_kappa calculates the quadratic weighted kappa
       value, which is a measure of inter-rater agreement between two raters
       that provide discrete numeric ratings.  Potential values range from -1
       (representing complete disagreement) to 1 (representing complete
       agreement).  A kappa value of 0 is expected if all agreement is due to
       chance.
       quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
       each correspond to a list of integer ratings.  These lists must have the
       same length.
       The ratings should be integers, and it is assumed that they contain
       the complete range of possible ratings.
       quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
       is the minimum possible rating, and max_rating is the maximum possible
       rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def import_data(p_data_path):
    """
        Function to import the data and return a dataframe
        :return df Dataframe containing the imported data
    """
    print('Entered import_data')
    df = pd.read_csv(p_data_path + '/train/train.csv')
    train_id = df['PetID']

    sentiment_mag = []
    sentiment_score = []
    for pet in train_id:
        try:
            with open(p_data_path + '/train_sentiment/' + pet + '.json', 'r', encoding='UTF-8') as f:
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
            with open(p_data_path + '/train_metadata/' + pet + '-1.json', 'r', encoding='UTF-8') as f:
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

    df = description_feat(df)

    return df


def description_feat(df):
    print('------- Build Description features -------')
    print('Vectorize Descriptions')
    descriptions = df.Description.fillna("no_desc").values
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b', use_idf=True)
    X = vectorizer.fit_transform(list(descriptions))
    print('X:', X.shape)

    print('SVD to reduce dimensionality')
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

    kappa_scorer = make_scorer(quadratic_weighted_kappa)
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
    skf = StratifiedKFold(n_splits=cv)
    i = 0
    cv_score = []
    model_list = []
    feature_importances = []

    for train_index, test_index in skf.split(X, y):
        i += 1
        print("training fold {} of {}".format(i, cv))
        X_train, X_test = np.array(X)[train_index, :], np.array(X)[test_index, :]
        y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

        model_copy = clone(model)
        model_copy.fit(X_train, y_train)
        score = quadratic_weighted_kappa(y_test, model_copy.predict(X_test))
        print("Score:", score)
        cv_score.append(score)
        model_list.append(model_copy)
        print("------- Feature_Importance of model {} -------".format(i))
        print(model_copy.feature_importances_)
        print(X.columns)

        feature_importances.append({i, zip(model_copy.feature_importances_, X.columns)})

    print("Mean cv Score", np.mean(cv_score))

    return model_list, feature_importances


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

    lgbm_list, feature_importances  = train_and_run_cv(lgbm, X, y, 5)

    # lgbm = do_grid_search(X, y)

    # df_test['lgbm_pred'] = lgbm.predict(df_test[feature_list])
    df_test['lgbm_pred'] = pred_ensemble(lgbm_list, df_test[feature_list])
    # df_test['lgbm_pred'] = np.round(df_test['lgbm_pred'], 0)

    opt_round = OptimizedRounder()

    opt_round.fit(df_test['lgbm_pred'], df_test['AdoptionSpeed'])
    df_test['lgbm_opt_pred'] = opt_round.predict(df_test['lgbm_pred'], opt_round.coefficients())

    lgbm_kappa = quadratic_weighted_kappa(df_test['AdoptionSpeed'], df_test['lgbm_pred'])
    lgbm_opt_kappa = quadratic_weighted_kappa(df_test['AdoptionSpeed'], df_test['lgbm_opt_pred'])

    print('Model tested! Quadratic Weighted Kappa: ' + str(lgbm_kappa))
    print('Optimized Model tested! QWK: ' + str(lgbm_opt_kappa))

    print('Coef: '+str(opt_round.coefficients()))

    df_test[['PetID', 'lgbm_opt_pred']].to_csv('submission.csv', index=False, header=['PetID', 'AdoptionSpeed'])

    # feature_imp = pd.DataFrame(sorted(feature_importances), columns=['Value', 'Feature'])

    print(list(feature_importances))


if __name__ == '__main__':
    main(sys.argv[1:])
