from joblib import load
from tensorflow import keras
import sklearn
from flask import Flask
from flask import request
import os
import library
from view import fpage
import pandas as pd
import numpy as np

the_transformer = library.nba_transformer
file_store = 'models/'
xgb_model = load(file_store+'xgb_model.joblib')
logreg_model = load(file_store+'logreg_model.joblib')
knn_model = load(file_store+'knn_model.joblib')
ann_model = keras.models.load_model(file_store+'ann_model')

test_df = pd.read_csv(f'test_df.csv')
logreg_thresholds = pd.read_csv(f'logreg_thresholds.csv').round(2)
knn_thresholds = pd.read_csv(f'knn_thresholds.csv').round(2)
xgb_thresholds = pd.read_csv(f'xgb_thresholds.csv').round(2)
ann_thresholds = pd.read_csv(f'ann_thresholds.csv').round(2)

def handle_data(columns, test_df, the_transformer):

    #massage data coming from user - this is code you need to change to fit your new dataset
    wins = np.nan
    if columns['wins'].isdigit() and int(columns['wins'])>0:
        wins =  int(columns['wins'])  #looks ok

    losses = np.nan
    if columns['losses'].isdigit() and int(columns['losses'])>=0:
        losses =  int(columns['losses'])

    ortg = np.nan 
    if columns['ortg'].isdigit() and int(columns['ortg'])>0:
        ortg =  int(columns['ortg'])
    drtg = np.nan
    if columns['drtg'].isdigit() and int(columns['drtg'])>0:
        drtg =  int(columns['drtg'])
    pace = np.nan 
    if columns['pace'].isdigit() and int(columns['pace'])>0:
        pace =  int(columns['pace'])
    srs = np.nan 
    if columns['srs'].isdigit() and int(columns['srs'])>0:
        srs =  int(columns['srs'])
    finish = np.nan if columns['finish']=='unknown' else int(columns['finish'])
    league = np.nan if columns['league']=='unknown' else columns['league']

    row = dict(W=wins, L=losses, ORtg=ortg, DRtg=drtg, Pace=pace, SRS=srs, Finish=finish, League=league)  #33.0, 'Female', 'C2', 'Southampton', 0.0, 26.0

    #end massaging - you should be able to use the code below as is

    #now add on your new row so can run pipeline
    n = len(test_df)
    test_extended = test_df.copy()  #don't mess up original
    test_extended.loc[n] = np.nan  #add blank row

    #fill in values we have from user
    for k,v in row.items():
        test_extended.loc[n, k] = v

    #run pipeline
    test_transformed = the_transformer.fit_transform(test_extended)

    #grab added row
    new_row = test_transformed.to_numpy()[-1:]

    #get predictions
    yhat_xgb, yhat_knn, yhat_logreg, yhat_ann = get_prediction(new_row)  #predict on last row that tacked on.
    return yhat_xgb, yhat_knn, yhat_logreg, yhat_ann

def get_prediction(row):
    assert row.shape[0]==1, f'Expecting numpy array but got {row}'
    #XGBoost
    xgb_raw = xgb_model.predict_proba(row)  #predict last row, we just tacked on
    yhat_xgb = xgb_raw[:,1]

    #KNN
    knn_raw = knn_model.predict_proba(row)
    yhat_knn = knn_raw[:,1]

    #logreg
    logreg_raw = logreg_model.predict_proba(row)
    yhat_logreg = logreg_raw[:,1]


    #ANN
    yhat_ann = ann_model.predict(row)[:,0]

    return [yhat_xgb, yhat_knn, yhat_logreg, yhat_ann]

xgb_table = xgb_thresholds.to_html(index=False, justify='center', col_space=80).replace('<td>', '<td align="center">')
logreg_table = logreg_thresholds.to_html(index=False, justify='center', col_space=80).replace('<td>', '<td align="center">')
knn_table = knn_thresholds.to_html(index=False, justify='center', col_space=80).replace('<td>', '<td align="center">')
ann_table = ann_thresholds.to_html(index=False, justify='center', col_space=80).replace('<td>', '<td align="center">')

fpage = fpage.replace('%xgb_table%', xgb_table)
fpage = fpage.replace('%logreg_table%', logreg_table)
fpage = fpage.replace('%knn_table%', knn_table)
fpage = fpage.replace('%ann_table%', ann_table)

def create_page(page, **fillers):
  new_page = page[:]  #copy
  for k,v in fillers.items():
    new_page = new_page.replace(f'%{str(k)}%', str(v))
  return new_page

os.environ["FLASK_ENV"] = "production"
app = Flask(__name__)
port = 5000

# Define Flask routes
@app.route("/nba-mlops")
#This function called when user first enters url into browser
def home():
    return create_page(fpage)

@app.route('/nba-mlops/data', methods = ['POST'])
#This function called when user hits Evaluate button
def data():
  form_data = request.form
  print(form_data.to_dict())
  yhat_xgb, yhat_knn, yhat_logreg, yhat_ann = handle_data(form_data.to_dict(),test_df, the_transformer)  #calling my own function here
  ensemble = (yhat_xgb[0]+yhat_knn[0]+yhat_logreg[0]+yhat_ann[0])/4.0
  xgb = np.round(yhat_xgb[0], 2)
  knn = np.round(yhat_knn[0], 2)
  logreg = np.round(yhat_logreg[0], 2)
  ann = np.round(yhat_ann[0], 2)
  ensemble = np.round(ensemble, 2)
  return create_page(fpage, xgb=xgb, knn=knn, logreg=logreg, ann=ann, ensemble=ensemble)  #return specs for new page with answer filled in

if __name__ == "__main__":
    app.run(host='0.0.0.0')