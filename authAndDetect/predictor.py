from plantDis import settings
from . my_models import *
import pickle
import numpy as np
from pathlib import Path

apple_dict={0 :'Apple___healthy', 1: 'Apple___Apple_scab', 2: 'Apple___Black_rot', 3: 'Apple___Cedar_apple_rust'}
corn_dict={0: 'Corn_(maize)___healthy',
	1: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
	2: 'Corn_(maize)__Common_rust',
	3: 'Corn_(maize)___Northern_Leaf_Blight'}

grapes_dict={0 : 'Grape___healthy',
	1 : 'Grape___Black_rot',
	2 : 'Grape___Esca_(Black_Measles)',
	3 : 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'}

potato_dict={0: 'Potato___healthy',
	1: 'Potato___Early_blight',
	2: 'Potato___Late_blight'}

tomato_dict={0 : 'Tomato___healthy',
	1 : 'Tomato___Bacterial_spot',
	2 : 'Tomato___Early_blight',
	3 : 'Tomato___Late_blight',
	4 : 'Tomato___Leaf_Mold',
	5 : 'Tomato___Septoria_leaf_spot',
	6 : 'Tomato___Spider_mites Two-spotted_spider_mite',
	7 : 'Tomato___Target_Spot',
	8 : 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
	9 : 'Tomato___Tomato_mosaic_virus'}


def get_model_and_result(plant,model_name,data):
    model_suffix=plant+'_'+model_name+'.pkl'
    print("====model_suffix=====",model_suffix)
    model_path=str(Path.joinpath(settings.MODELS_ROOT,model_suffix))
    model=pickle.load(open(model_path, 'rb'))
    return predict(model,data,plant)

def predict(model,feature_vector,plant):
    processed_vector = np.array(feature_vector).reshape(1, -1)
    output = model.predict(processed_vector)
    output = int(output)
    dict_name=plant+'_dict'
    label_dict = globals()[dict_name]
    output = label_dict[output]
    return output
