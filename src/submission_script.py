__author__ = "Robin Brecht, Philipp Paraguya"
__credits__ = ["Robin Brecht", "Philipp Paraguya"]

import os
import json
import pandas as pd
import sys
pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score,  make_scorer
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

def squared_cohen_kappa(y1, y2):
    return cohen_kappa_score(y1,y2)**2



def main(argv):
    """
    Diese Funktion ist der Einstiegspunkt für dieses Projekt
    """

    DATA_PATH = argv[0]

    df = pd.read_csv(DATA_PATH + '/train/train.csv')
    train_id = df['PetID']

    sentiment_mag = []
    sentiment_score = []
    for pet in train_id:
        try:
            with open(DATA_PATH + '/train_sentiment/' + pet + '.json', 'r', encoding='UTF-8') as f:
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
            with open(DATA_PATH + '/train_metadata/' + pet + '-1.json', 'r', encoding='UTF-8') as f:
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

    df.head()

    df_train, df_test = train_test_split(df, test_size=0.2)

    drop_feat_list = ['AdoptionSpeed', 'Name', 'RescuerID', 'Description', 'PetID', 'label_description']

    feature_list = list(df.columns)
    feature_list = [x for x in feature_list if x not in drop_feat_list]

    X = df_train[feature_list]
    y = df_train['AdoptionSpeed'].values

    lgbm = LGBMClassifier(objective='multiclass', random_state=5)

    lgbm.fit(X, y)
    df_test['lgbm_pred'] = lgbm.predict(df_test[feature_list])

    lgbm_kappa = squared_cohen_kappa(df_test['AdoptionSpeed'], df_test['lgbm_pred'])

    print('Model tested! Squared Cohen Kappa: ' + str(lgbm_kappa))

    df_test[['PetID','lgbm_pred']].to_csv('submission.csv', index=False, header=['PetID','AdoptionSpeed'])

if __name__ == '__main__':
    main(sys.argv[1:])
