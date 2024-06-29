import streamlit as st
from keras.models import Sequential,model_from_json 
import pandas as pd 

# load json and create model

# Write the file name of the model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")



# insert / provide input data here
prediction_df = pd.DataFrame([{"statuses_count" : 1,
    "followers_count":554,
    "friends_count":534,
    "favourites_count":0,
    "lang_num":1,
    "listed_count":0,
    "geo_enabled":1,
    "profile_use_background_image":1}])

#print(prediction_df)
st.write(prediction_df)

prediction = loaded_model.predict(prediction_df)
prediction = prediction[0]

st.write('Prediction\n',prediction)
#print('Prediction\n',prediction)
# print('\nThresholded output\n',(prediction>0.5)*1)
if prediction > 0.5:
    st.write("fake profile")
    #print("fake profile")
else:
    st.write("real profile")
    # print("real profile")



