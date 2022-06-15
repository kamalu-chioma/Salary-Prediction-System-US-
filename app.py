#load core pkgs
import streamlit as st

#EDA pkgs
import pandas as pd 
import numpy as np 

#Data viz pkgs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
#import sklearn.neighbors._dist_metrics
#ML pkgs
import joblib,os
from PIL import Image
import sqlite3
import datetime



def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value


#get keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key

#load models
def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model



#creating a class
class Monitor(object):
	"""docstring for Monitor"""

	conn = sqlite3.connect("data.db")
	c = conn.cursor()

	def __init__(self, age=None,workclass=None,fnlwgt=None,education=None,education_num=None,marital_status=None,occupation=None,relationship=None,race=None,sex=None,capital_gain=None,capital_loss=None,hours_per_week=None,native_country=None,predicted_class = None,model_class = None,time_of_prediction = None):
		super(Monitor, self).__init__()
		self.age = age
		self.workclass = workclass
		self.fnlwgt = fnlwgt
		self.education = education
		self.education_num = education_num
		self.marital_status = marital_status
		self.occupation = occupation
		self.relationship = relationship
		self.race = race
		self.sex = sex
		self.capital_gain = capital_gain
		self.capital_loss = capital_loss
		self.hours_per_week = hours_per_week
		self.native_country = native_country
		self.predicted_class = predicted_class
		self.model_class = model_class
		self.time_of_prediction = time_of_prediction

	def __repr__(self):
		# return "Monitor(age ={self.age},workclass ={self.workclass},fnlwgt ={self.fnlwgt},education ={self.education},education_num ={self.education_num},marital_status ={self.marital_status},occupation ={self.occupation},relationship ={self.relationship},race ={self.race},sex ={self.sex},capital_gain ={self.capital_gain},capital_loss ={self.capital_loss},hours_per_week ={self.hours_per_week},native_country ={self.native_country},predicted_class ={self.predicted_class},model_class ={self.model_class})".format(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class)
		"Monitor(age = {self.age},workclass = {self.workclass},fnlwgt = {self.fnlwgt},education = {self.education},education_num = {self.education_num},marital_status = {self.marital_status},occupation = {self.occupation},relationship = {self.relationship},race = {self.race},sex = {self.sex},capital_gain = {self.capital_gain},capital_loss = {self.capital_loss},hours_per_week = {self.hours_per_week},native_country = {self.native_country},predicted_class = {self.predicted_class},model_class = {self.model_class})".format(self=self)

	def create_table(self):
		self.c.execute("CREATE TABLE IF NOT EXISTS predictiontable(age NUMERIC,workclass NUMERIC,fnlwgt NUMERIC,education NUMERIC,education_num NUMERIC,marital_status NUMERIC,occupation NUMERIC,relationship NUMERIC,race NUMERIC,sex NUMERIC,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,native_country NUMERIC,predicted_class NUMERIC,model_class TEXT)")

	def add_data(self):
		self.c.execute("INSERT INTO predictiontable(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,predicted_class,model_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class))
		self.conn.commit()

	def view_all_data(self):
		self.c.execute('SELECT * FROM predictiontable')
		data = self.c.fetchall()
		# for row in data:
		# 	print(row)
		return data

	def view_all_data(self):
		self.c.execute('SELECT * FROM predictiontable')
		data = self.c.fetchall()
		# for row in data:
		# 	print(row)
		return data







def main():
	""" Salary Predictor with ML"""
	st.title("Salary Predictor")
	activity = ["EDA","Predictions","Metrics","Countries"]
	choice = st.sidebar.selectbox('Choose An Activity',activity)
	#load file
	df = pd.read_csv("datasets/cleaned_salary_prediction_dataset_numeric.csv",usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])



	#EDA
	if choice == "EDA":
		st.subheader("EDA Section")
		st.text("Exploratory Data Analysis")
		#preview
		if st.checkbox("Preview Dataset"):
			number = int(st.number_input("Number to show"))
			st.dataframe(df.head(number))
		

		#show columns/rows
		if st.button("Column Names"):
			st.write(df.columns)

		#description
		if st.checkbox("Show Description"):
			st.write(df.describe())
		
		#shape
		if st.checkbox("Show Shape of Dataset"):
			st.write(df.shape)
			data_dim = st.radio("Show Dimensions by",("Rows","Columns"))
			
			if data_dim=="Rows":
				#st.text("Number of Rows")
				st.write("Number of Rows: ",df.shape[0])

			elif data_dim == "Columns":
				#st.text("Number of Columns")
				st.write("Number of Columns: ",df.shape[1])

			else:
				st.write("df.shape")
		

		#selections
		if st.checkbox("Select Columns to Show"):
			all_columns = df.columns.to_list()
			selected_columns = st.multiselect("Select Columns",all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)


		if st.checkbox('Select Row(s) Index to Show'):
			Selected_index = st.multiselect("Select Rows",df.head(20).index)
			selected_rows = df.loc[Selected_index]
			st.dataframe(selected_rows)
		
		#value count
		if st.button("Value Counts"):
			st.text("Value Counts by Class")
			st.write(df.iloc[:,-1].value_counts())
		

		#PLOTS
		#if st.checkbox("Show Correlation Plot[Matplotlib]"):
			#plt.matshow(df.corr())
			#fig,ax = plt.subplots()
			#(df.corr(),ax=ax)
			#st.write(fig)


		if st.checkbox("Show Correlation Plot[seaborn]"):
			#plt.matshow(df.corr())
			fig,ax = plt.subplots()
			sns.heatmap(df.corr(),ax=ax,annot = False)
			st.write(fig)
	

	#PREDICTION
	if choice == "Predictions":
		
		st.subheader("Prediction Section")
		
		d_workclass = {' Federal-gov': 0, ' Local-gov': 1, ' Never-worked': 2, ' Private': 3, ' Self-emp-inc': 4, ' Self-emp-not-inc': 5, ' State-gov': 6, ' Without-pay': 7}

		d_education = {' 10th': 0, ' 11th': 1, ' 12th': 2, ' 1st-4th': 3, ' 5th-6th': 4, ' 7th-8th': 5, ' 9th': 6, ' Assoc-acdm': 7, ' Assoc-voc': 8, ' Bachelors': 9, ' Doctorate': 10, ' HS-grad': 11, ' Masters': 12, ' Preschool': 13, ' Prof-school': 14, ' Some-college': 15}

		d_marital_status = {' Divorced': 0, ' Married-AF-spouse': 1, ' Married-civ-spouse': 2, ' Married-spouse-absent': 3, ' Never-married': 4, ' Separated': 5, ' Widowed': 6}
		
		d_occupation = {' Adm-clerical': 0, ' Armed-Forces': 1, ' Craft-repair': 2, ' Exec-managerial': 3, ' Farming-fishing': 4, ' Handlers-cleaners': 5, ' Machine-op-inspct': 6, ' Other-service': 7, ' Priv-house-serv': 8, ' Prof-specialty': 9, ' Protective-serv': 10, ' Sales': 11, ' Tech-support': 12, ' Transport-moving': 13}

		d_relationship = {' Husband': 0, ' Not-in-family': 1, ' Other-relative': 2, ' Own-child': 3, ' Unmarried': 4, ' Wife': 5}

		d_race = {' Amer-Indian-Eskimo': 0, ' Asian-Pac-Islander': 1, ' Black': 2, ' Other': 3, ' White': 4}

		d_sex = {' Female': 0, ' Male': 1}

		d_native_country = {' Cambodia': 0, ' Canada': 1, ' China': 2, ' Columbia': 3, ' Cuba': 4, ' Dominican-Republic': 5, ' Ecuador': 6, ' El-Salvador': 7, ' England': 8, ' France': 9, ' Germany': 10, ' Greece': 11, ' Guatemala': 12, ' Haiti': 13, ' Holand-Netherlands': 14, ' Honduras': 15, ' Hong': 16, ' Hungary': 17, ' India': 18, ' Iran': 19, ' Ireland': 20, ' Italy': 21, ' Jamaica': 22, ' Japan': 23, ' Laos': 24, ' Mexico': 25, ' Nicaragua': 26, ' Outlying-US(Guam-USVI-etc)': 27, ' Peru': 28, ' Philippines': 29, ' Poland': 30, ' Portugal': 31, ' Puerto-Rico': 32, ' Scotland': 33, ' South': 34, ' Taiwan': 35, ' Thailand': 36, ' Trinadad&Tobago': 37, ' United-States': 38, ' Vietnam': 39, ' Yugoslavia': 40}

		d_class = {'<=50K': 0, '>50K': 1}


		#ML Aspect User input
		age = st.slider("Select Age",17,90)
		workclass = st.selectbox("Select Work Class",tuple(d_workclass.keys()))
		fnlwgt = st.number_input("Enter FNLWGT",12285,1484705)
		education = st.selectbox("Select Education",tuple(d_education.keys()))
		education_num = st.slider("Select years of Education",1,16)
		marital_status = st.selectbox("Select Marital Status",tuple(d_marital_status.keys()))
		occupation = st.selectbox("Select Occupation",tuple(d_occupation.keys()))
		relationship = st.selectbox("Select Relationship",tuple(d_relationship.keys()))
		race = st.selectbox("Select Race",tuple(d_race.keys()))
		sex = st.radio("Select Gender",tuple(d_sex.keys()))
		capital_gain = st.number_input("Capital Gain",0,99999)
		capital_loss = st.number_input("Capital loss",0,4356)
		hours_per_week = st.number_input("Hours Per Week",1,99)
		native_country = st.selectbox("Select Native Country",tuple(d_native_country.keys()))


		#User Input
		k_workclass = get_value(workclass,d_workclass)
		k_education = get_value(education,d_education)
		k_marital_status = get_value(marital_status,d_marital_status)
		k_occupation = get_value(occupation,d_occupation)
		k_relationship = get_value(relationship,d_relationship)
		k_race = get_value(race,d_race)
		k_sex = get_value(sex,d_sex)
		k_native_country = get_value(native_country,d_native_country)




		#Result of user input
		selected_options = [age,workclass,fnlwgt,
							education_num,education,marital_status,
							occupation,relationship,race,sex,native_country,
							capital_gain,capital_loss,hours_per_week]


		vectorized_result = [age,k_workclass,fnlwgt,
							education_num,k_education,k_marital_status,
							k_occupation,k_relationship,k_race,k_sex,
							k_native_country,capital_gain,capital_loss,
							hours_per_week]

		sample_data = np.array(vectorized_result).reshape(1,-1)					

		st.info(selected_options)
		pretiffied_result = {"age":age,"workclass":workclass,"fnlwgt":fnlwgt,
							"education_num":education_num,
							"education":education,
							"marital_status":marital_status,
							"occupation":occupation,
							"relationship":relationship,
							"race":race,
							"sex":sex,
							"native_country":native_country,
							"capital_gain":capital_gain,
							"capital_loss":capital_loss,
							"hours_per_week":hours_per_week}
		st.json(pretiffied_result)
		st.write(vectorized_result)
		

		#MAKE PREDICTION
		st.subheader("Prediction")
		if st.checkbox("Make Prediction"):
			all_ml_dict = {'DecisionTree Classifier':"DT",'Bagging Classifier':"BC",'Logistic Regression':"LR"}
				                                      
				

		#MODEL SELECTION OPTION
			model_choice = st.selectbox("Model Choice",list(all_ml_dict.keys()))
			#prediction_label = {'<=50K': 0, '>50K': 1}
			prediction_label = {' Earns $50,000 or less': 0, ' Earns more than $50,000': 1}

			 #Earns $50,000 or less,Earns more than $50,000"
			if st.button("Predict"):

				if model_choice == "Bagging Classifier":
					model_predictor = load_prediction_models("salary_prediction_models/salary_predictions_BaggingClassifier_model.pkl")
					prediction = model_predictor.predict(sample_data)
					#st.write(prediction)

				elif model_choice == "DecisionTree Classifier":
					model_predictor = load_prediction_models("salary_prediction_models/salary_predictions_DecisionTreeClassifier_model.pkl")
					prediction = model_predictor.predict(sample_data)
								
				elif model_choice == "Logistic Regression":
					model_predictor = load_prediction_models("salary_prediction_models/salary_predictions_LogisticRegression_model.pkl")
					prediction = model_predictor.predict(sample_data)
					
				final_result = get_key(prediction,prediction_label)
				model_class = model_choice
				#time_of_prediction = datetime.datetime.now()
				monitor =Monitor(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,final_result,model_class)
				monitor.create_table()
				monitor.add_data()
				st.success("Predicted Salary as :: {}".format(final_result))



	
	#METRICS
	if choice == 'Metrics':
		st.subheader("Metrics Section")
		cnx = sqlite3.connect('data.db')
		mdf = pd.read_sql_query("SELECT * FROM predictiontable",cnx)
		st.dataframe(mdf)

if __name__ == '__main__':
	main()