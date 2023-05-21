{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1d3373ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import all the app dependencies\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import sklearn\n",
    "import streamlit as st\n",
    "import joblib\n",
    "import matplotlib\n",
    "from IPython import get_ipython\n",
    "from PIL import Image\n",
    "\n",
    "# load the encoder and model object\n",
    "model = joblib.load(\"rta_model_deploy3.joblib\")\n",
    "encoder = joblib.load(\"ordinal_encoder2.joblib\")\n",
    "\n",
    "st.set_option('deprecation.showPyplotGlobalUse', False)\n",
    "\n",
    "# 1: serious injury, 2: Slight injury, 0: Fatal Injury\n",
    "\n",
    "st.set_page_config(page_title=\"Accident Severity Prediction App\",\n",
    "        page_icon=\"🚧\", layout=\"wide\")\n",
    "\n",
    "#creating option list for dropdown menu\n",
    "options_day = ['Sunday', \"Monday\", \"Tuesday\", \"Wednesday\", \"Thursday\", \"Friday\", \"Saturday\"]\n",
    "options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']\n",
    "\n",
    "# number of vehicle involved: range of 1 to 7\n",
    "# number of casualties: range of 1 to 8\n",
    "# hour of the day: range of 0 to 23\n",
    "\n",
    "options_types_collision = ['Vehicle with vehicle collision','Collision with roadside objects',\n",
    "              'Collision with pedestrians','Rollover','Collision with animals',\n",
    "              'Unknown','Collision with roadside-parked vehicles','Fall from vehicles',\n",
    "              'Other','With Train']\n",
    "\n",
    "options_sex = ['Male','Female','Unknown']\n",
    "\n",
    "options_education_level = ['Junior high school','Elementary school','High school',\n",
    "              'Unknown','Above high school','Writing & reading','Illiterate']\n",
    "\n",
    "options_services_year = ['Unknown','2-5yrs','Above 10yr','5-10yrs','1-2yr','Below 1yr']\n",
    "\n",
    "options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',\n",
    "    ' Industrial areas', 'School areas', ' Recreational areas',\n",
    "    ' Outside rural areas', ' Hospital areas', ' Market areas',\n",
    "    'Rural village areas', 'Unknown', 'Rural village areasOffice areas',\n",
    "    'Recreational areas']\n",
    "\n",
    "# features list\n",
    "features = ['Number_of_vehicles_involved','Number_of_casualties','Hour_of_Day','Type_of_collision','Age_band_of_driver','Sex_of_driver',\n",
    "    'Educational_level','Service_year_of_vehicle','Day_of_week','Area_accident_occured']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c4e4203",
   "metadata": {},
   "source": [
    "The code first imports the necessary libraries and modules required for the app; these are the libraries that we stated in our requirements.txt file.\n",
    "\n",
    "It then loads two pre-trained objects, an encoder and a machine learning model, which will be used to encode categorical variables and predict the accident severity, respectively. These are the models we downloaded in Step 3.\n",
    "\n",
    "We then set some Streamlit options and configurations, including the page title, icon, and layout, and defines some dropdown menu options for various features related to the accident, such as the day of the week, age band of the driver, type of collision, and area where the accident occurred.\n",
    "\n",
    "Finally, the code creates a list of features that will be used as input to the machine learning model. This list includes the number of vehicles involved, the number of casualties, the hour of the day, the type of collision, the age band of the driver, the sex of the driver, the educational level, the service year of the vehicle, the day of the week, and the area where the accident occurred."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1c66ad89",
   "metadata": {},
   "source": [
    "## Step 6: App.py - User Input & Model Prediction\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86319cc1",
   "metadata": {},
   "source": [
    "Now that we’ve defined the inputs to be taken from the user, let’s see how we can define the main() function to develop UI that will be rendered on the front end.\n",
    "\n",
    "Copy the code below and paste it into app.py.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7ebfd2dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Give a title to web app using html syntax\n",
    "st.markdown(\"<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>\", unsafe_allow_html=True)\n",
    "\n",
    "# define a main() function to take inputs from user in form based approach\n",
    "def main():\n",
    "    with st.form(\"road_traffic_severity_form\"):\n",
    "       st.subheader(\"Please enter the following inputs:\")\n",
    "        \n",
    "       No_vehicles = st.slider(\"Number of vehicles involved:\",1,7, value=0, format=\"%d\")\n",
    "       No_casualties = st.slider(\"Number of casualties:\",1,8, value=0, format=\"%d\")\n",
    "       Hour = st.slider(\"Hour of the day:\", 0, 23, value=0, format=\"%d\")\n",
    "       collision = st.selectbox(\"Type of collision:\",options=options_types_collision)\n",
    "       Age_band = st.selectbox(\"Driver age group?:\", options=options_age)\n",
    "       Sex = st.selectbox(\"Sex of the driver:\", options=options_sex)\n",
    "       Education = st.selectbox(\"Education of driver:\",options=options_education_level)\n",
    "       service_vehicle = st.selectbox(\"Service year of vehicle:\", options=options_services_year)\n",
    "       Day_week = st.selectbox(\"Day of the week:\", options=options_day)\n",
    "       Accident_area = st.selectbox(\"Area of accident:\", options=options_acc_area)\n",
    "        \n",
    "       submit = st.form_submit_button(\"Predict\")\n",
    "\n",
    "# encode using ordinal encoder and predict\n",
    "    if submit:\n",
    "       input_array = np.array([collision,\n",
    "                  Age_band,Sex,Education,service_vehicle,\n",
    "                  Day_week,Accident_area], ndmin=2)\n",
    "        \n",
    "       encoded_arr = list(encoder.transform(input_array).ravel())\n",
    "        \n",
    "       num_arr = [No_vehicles,No_casualties,Hour]\n",
    "       pred_arr = np.array(num_arr + encoded_arr).reshape(1,-1)        \n",
    "      \n",
    "# predict the target from all the input features\n",
    "       prediction = model.predict(pred_arr)\n",
    "        \n",
    "       if prediction == 0:\n",
    "           st.write(f\"The severity prediction is fatal injury⚠\")\n",
    "       elif prediction == 1:\n",
    "           st.write(f\"The severity prediction is serious injury\")\n",
    "       else:\n",
    "           st.write(f\"The severity prediction is slight injury\")\n",
    "        \n",
    "       st.write(\"Developed By: Avi kumar Talaviya\")\n",
    "       st.markdown(\"\"\"Reach out to me on: [Twitter](https://twitter.com/avikumart_) |\n",
    "       [Linkedin](https://www.linkedin.com/in/avi-kumar-talaviya-739153147/) |\n",
    "       [Kaggle](https://www.kaggle.com/avikumart) \n",
    "       \"\"\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47371012",
   "metadata": {},
   "source": [
    "## Step 7: App.py - Webpage Aesthetics\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e70613d5",
   "metadata": {},
   "outputs": [
    {
     "ename": "RuntimeError",
     "evalue": "Runtime hasn't been created!",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\streamlit\\elements\\image.py:361\u001b[0m, in \u001b[0;36mimage_to_url\u001b[1;34m(image, width, clamp, channels, output_format, image_id)\u001b[0m\n\u001b[0;32m    360\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 361\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28;43mopen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mimage\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mrb\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[0;32m    362\u001b[0m         image_data \u001b[38;5;241m=\u001b[39m f\u001b[38;5;241m.\u001b[39mread()\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'banner-picture.jpeg'",
      "\nDuring handling of the above exception, another exception occurred:\n",
      "\u001b[1;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
      "Input \u001b[1;32mIn [7]\u001b[0m, in \u001b[0;36m<cell line: 2>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      1\u001b[0m a,b,c \u001b[38;5;241m=\u001b[39m st\u001b[38;5;241m.\u001b[39mcolumns([\u001b[38;5;241m0.2\u001b[39m,\u001b[38;5;241m0.6\u001b[39m,\u001b[38;5;241m0.2\u001b[39m])\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m b:\n\u001b[1;32m----> 3\u001b[0m  \u001b[43mst\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mimage\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mbanner-picture.jpeg\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43muse_column_width\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m)\u001b[49m\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# description about the project and code files       \u001b[39;00m\n\u001b[0;32m      7\u001b[0m st\u001b[38;5;241m.\u001b[39msubheader(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m🧾Description:\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\streamlit\\runtime\\metrics_util.py:332\u001b[0m, in \u001b[0;36mgather_metrics.<locals>.wrapped_func\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    330\u001b[0m         _LOGGER\u001b[38;5;241m.\u001b[39mdebug(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mFailed to collect command telemetry\u001b[39m\u001b[38;5;124m\"\u001b[39m, exc_info\u001b[38;5;241m=\u001b[39mex)\n\u001b[0;32m    331\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 332\u001b[0m     result \u001b[38;5;241m=\u001b[39m non_optional_func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    333\u001b[0m \u001b[38;5;28;01mfinally\u001b[39;00m:\n\u001b[0;32m    334\u001b[0m     \u001b[38;5;66;03m# Activate tracking again if command executes without any exceptions\u001b[39;00m\n\u001b[0;32m    335\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m ctx:\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\streamlit\\elements\\image.py:169\u001b[0m, in \u001b[0;36mImageMixin.image\u001b[1;34m(self, image, caption, width, use_column_width, clamp, channels, output_format)\u001b[0m\n\u001b[0;32m    166\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m StreamlitAPIException(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mImage width must be positive.\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m    168\u001b[0m image_list_proto \u001b[38;5;241m=\u001b[39m ImageListProto()\n\u001b[1;32m--> 169\u001b[0m \u001b[43mmarshall_images\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    170\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdg\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_get_delta_path_str\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    171\u001b[0m \u001b[43m    \u001b[49m\u001b[43mimage\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    172\u001b[0m \u001b[43m    \u001b[49m\u001b[43mcaption\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    173\u001b[0m \u001b[43m    \u001b[49m\u001b[43mwidth\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    174\u001b[0m \u001b[43m    \u001b[49m\u001b[43mimage_list_proto\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    175\u001b[0m \u001b[43m    \u001b[49m\u001b[43mclamp\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    176\u001b[0m \u001b[43m    \u001b[49m\u001b[43mchannels\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    177\u001b[0m \u001b[43m    \u001b[49m\u001b[43moutput_format\u001b[49m\u001b[43m,\u001b[49m\n\u001b[0;32m    178\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[0;32m    179\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mdg\u001b[38;5;241m.\u001b[39m_enqueue(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mimgs\u001b[39m\u001b[38;5;124m\"\u001b[39m, image_list_proto)\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\streamlit\\elements\\image.py:536\u001b[0m, in \u001b[0;36mmarshall_images\u001b[1;34m(coordinates, image, caption, width, proto_imgs, clamp, channels, output_format)\u001b[0m\n\u001b[0;32m    533\u001b[0m         is_svg \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m\n\u001b[0;32m    535\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m is_svg:\n\u001b[1;32m--> 536\u001b[0m     proto_img\u001b[38;5;241m.\u001b[39murl \u001b[38;5;241m=\u001b[39m \u001b[43mimage_to_url\u001b[49m\u001b[43m(\u001b[49m\n\u001b[0;32m    537\u001b[0m \u001b[43m        \u001b[49m\u001b[43mimage\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mwidth\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mclamp\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mchannels\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43moutput_format\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mimage_id\u001b[49m\n\u001b[0;32m    538\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\streamlit\\elements\\image.py:371\u001b[0m, in \u001b[0;36mimage_to_url\u001b[1;34m(image, width, clamp, channels, output_format, image_id)\u001b[0m\n\u001b[0;32m    368\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m mimetype \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m    369\u001b[0m     mimetype \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mapplication/octet-stream\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m--> 371\u001b[0m url \u001b[38;5;241m=\u001b[39m \u001b[43mruntime\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mget_instance\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[38;5;241m.\u001b[39mmedia_file_mgr\u001b[38;5;241m.\u001b[39madd(image, mimetype, image_id)\n\u001b[0;32m    372\u001b[0m caching\u001b[38;5;241m.\u001b[39msave_media_data(image, mimetype, image_id)\n\u001b[0;32m    373\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m url\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\streamlit\\runtime\\__init__.py:29\u001b[0m, in \u001b[0;36mget_instance\u001b[1;34m()\u001b[0m\n\u001b[0;32m     25\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mget_instance\u001b[39m() \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Runtime:\n\u001b[0;32m     26\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Return the singleton Runtime instance. Raise an Error if the\u001b[39;00m\n\u001b[0;32m     27\u001b[0m \u001b[38;5;124;03m    Runtime hasn't been created yet.\u001b[39;00m\n\u001b[0;32m     28\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m---> 29\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mRuntime\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43minstance\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[1;32m~\\anaconda3\\lib\\site-packages\\streamlit\\runtime\\runtime.py:146\u001b[0m, in \u001b[0;36mRuntime.instance\u001b[1;34m(cls)\u001b[0m\n\u001b[0;32m    142\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"Return the singleton Runtime instance. Raise an Error if the\u001b[39;00m\n\u001b[0;32m    143\u001b[0m \u001b[38;5;124;03mRuntime hasn't been created yet.\u001b[39;00m\n\u001b[0;32m    144\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    145\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mcls\u001b[39m\u001b[38;5;241m.\u001b[39m_instance \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[1;32m--> 146\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mRuntimeError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mRuntime hasn\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mt been created!\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m    147\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mcls\u001b[39m\u001b[38;5;241m.\u001b[39m_instance\n",
      "\u001b[1;31mRuntimeError\u001b[0m: Runtime hasn't been created!"
     ]
    }
   ],
   "source": [
    "a,b,c = st.columns([0.2,0.6,0.2])\n",
    "with b:\n",
    " st.image(\"banner-picture.jpeg\", use_column_width=True)\n",
    "\n",
    "\n",
    "# description about the project and code files       \n",
    "st.subheader(\"🧾Description:\")\n",
    "st.text(\"\"\"This data set is collected from Addis Ababa Sub-city police departments for master's research work. \n",
    "The data set has been prepared from manual records of road traffic accidents of the year 2017-20. \n",
    "All the sensitive information has been excluded during data encoding and finally it has 32 features and 12316 instances of the accident.\n",
    "Then it is preprocessed and for identification of major causes of the accident by analyzing it using different machine learning classification algorithms.\n",
    "\"\"\")\n",
    "\n",
    "st.markdown(\"Source of the dataset: [Click Here](https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591)\")\n",
    "\n",
    "st.subheader(\"🧭 Problem Statement:\")\n",
    "st.text(\"\"\"The target feature is Accident_severity which is a multi-class variable. \n",
    "The task is to classify this variable based on the other 31 features step-by-step by going through each day's task. \n",
    "The metric for evaluation will be f1-score\n",
    "\"\"\")\n",
    "\n",
    "st.markdown(\"Please find GitHub repository link of project: [Click Here](https://github.com/avikumart/Road-Traffic-Severity-Classification-Project)\")          \n",
    "  \n",
    "# run the main function        \n",
    "if __name__ == '__main__':\n",
    "  main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b6ceb112",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d9a321b6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e6b8c589",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb49dd51",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c685a26",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26bfd758",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
