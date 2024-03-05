from flask import Flask,render_template, request,url_for
import pickle
import numpy as np
import pandas as pd



app = Flask(__name__, template_folder='template')
model = pickle.load(open('breast_cancer_detection.pickle','rb'))

@app.route('/')

def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():

  input_features = [float(x) for x in request.form.values()]

  features_value = [np.array(input_features)]

  features_name = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area','mean_smoothness', 'mean_compactness', 'mean_concavity','mean_concave points', 'mean_symmetry', 'mean_fractal_dimension' 'radius_error', 'texture_error', 'perimeter_error', 'area error',  'smoothness error', 'compactness_error', 'concavity_error','concave_points_error', 'symmetry_error', 'fractal_dimension_error','worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']

  df = pd.DataFrame(features_value,columns=features_name)
  output = model.predict(df)

  if output == 0:
    res_val = "** You Have Breast Cancer"
  
  else:
    res_val="**No Breast Cancer"

if __name__ == 'main':
  app.run(debug=True)


