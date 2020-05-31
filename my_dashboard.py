# Import librairies.
import pandas as pd
import numpy as np
import pickle5 as pickle
import scipy

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import streamlit as st

###################### TITLE AND INTRO ######################

# Title.
st.title("Dashboard  intéractif : Détection de défaut de paiement")

# Header & Subheader.
st.header("Projet 7 - Implémenter un modèle de scoring")

# Markdown.
st.markdown('Autor : Louis Birenholz')
st.markdown("Ce dashboard expose la probabilité de défaut de paiement d'un client. Il permet une visualisation dynamique de certains indicateurs ainsi que le calcul d'une nouvelle probabilité de défaut.")

################### READ DATA ########################

# Loading data.
@st.cache
def load_data_brute():
	url = "https://raw.githubusercontent.com/Louis-bir/dashboard-project-credits.app/master/df_brute.csv"
	data = pd.read_csv(url, error_bad_lines=False)
	return data

df = load_data_brute()

@st.cache
def load_data():
	url = "https://raw.githubusercontent.com/Louis-bir/dashboard-project-credits.app/master/df_dashboard2.csv"
	data = pd.read_csv(url, error_bad_lines=False)
	return data

df_prob = load_data()

##################### IMPORTATION DU MODELE PREENTRAINE (LGBM) ###################

#with open(r"C:\Users\Louis\dashboard-project-credits.app\pretrainedLGBM.pickle", 'rb') as file : model_train = pickle.load(file)

##################### DASHBOARD ################################

# Side page.
st.sidebar.header("Id client")

id_client = st.sidebar.text_input('ID client', 'x')

if (id_client == 'x'):
	st.error('Merci de rentrer un ID client valide')

# Entrer de l'ID client et informations client.

df_id_client = df[df['SK_ID_CURR']==int(id_client)]
df_id_client2 = df_prob[df_prob['SK_ID_CURR']==int(id_client)]

if (df_id_client2.shape[0] == 0):
	st.error('Merci de rentrer un ID client valide')

df_info_client = df_id_client.loc[:,['AMT_CREDIT', 'NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']]
df_info_client.columns = ['AMT_CREDIT', 'NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'AGE', 'DAYS_EMPLOYED']
df_info_client.index = ['Information']
df_info_client['AGE'] = (-df_info_client['AGE']/365).round()
df_info_client['DAYS_EMPLOYED'] = -df_info_client['DAYS_EMPLOYED']

st.success('Informations client : ')
st.write(df_info_client)

int(df_info_client['AMT_CREDIT'])

# Sliders on side page.
st.sidebar.header("Sliders")
new_source1 = st.sidebar.slider('Source EXT 1', 0.0, 1.0, np.array(df_id_client['EXT_SOURCE_1'])[0], 0.001) 
new_source2 = st.sidebar.slider('Source EXT 2', 0.0, 1.0, np.array(df_id_client['EXT_SOURCE_2'])[0],  0.001) 
new_source3 = st.sidebar.slider('Source EXT 3', 0.0, 1.0, np.array(df_id_client['EXT_SOURCE_3'])[0],  0.001) 

# Predict.
mask_id = (df_prob['SK_ID_CURR'] == int(id_client))
client_to_predict = df_prob[mask_id]['Proba']
prediction = np.array(client_to_predict)[0].round(2)

st.subheader('Seuil de défaillance retenu : 0.23')
st.subheader('Probabilité de défaillance : {}'.format(prediction))

# Graph Income/Credit & Days Employed.
income_client = int(df_info_client['AMT_INCOME_TOTAL'])
percent_income_client = income_client*0.1

mask_1 = (df['AMT_INCOME_TOTAL'] <= income_client + percent_income_client)
mask_2 = (df['AMT_INCOME_TOTAL'] >= income_client - percent_income_client)

fig = make_subplots(rows=2, cols=1, subplot_titles=("Income & Credits","Days Employed"))

trace0 = go.Histogram(x=df[mask_1 & mask_2]['AMT_CREDIT'], 
					  name='Income & Credit',
					  xbins=dict(size=100000),
					  histnorm='percent',
					  marker_color='#EB89B5')

trace0_client = go.Scatter(x=[int(df_info_client['AMT_CREDIT']), int(df_info_client['AMT_CREDIT'])],
						  y=[0, 20], mode="lines", name="Client's credit",
                          line=go.scatter.Line(color="red"))

trace1 = go.Histogram(x=-df['DAYS_EMPLOYED'], 
					  name='Days Employed',
					  xbins=dict(size=500),
					  marker_color='#37AA9C',
					  histnorm='percent')

trace1_client = go.Scatter(x=[int(df_info_client['DAYS_EMPLOYED']), int(df_info_client['DAYS_EMPLOYED'])], mode="lines", y=[0, 14.5],
                          line=go.scatter.Line(color="black"), name="Client's days employed")

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)
fig.append_trace(trace0_client, 1, 1)
fig.append_trace(trace1_client, 2, 1)

fig.update_layout(height=640, width=850)

# Update yaxis properties
fig.update_yaxes(title_text="%", row=1, col=1)
fig.update_yaxes(title_text="%", row=2, col=1)

# Update xaxis properties
fig.update_xaxes(title_text="€", row=1, col=1)
fig.update_xaxes(title_text="Days", row=2, col=1)

st.plotly_chart(fig)


# Indicator
st.error('Autres indicateurs :')

mask_target1 = (df['TARGET'] == 1)
mask_target0 = (df['TARGET'] == 0)

data_source1 = [df[mask_target1]['EXT_SOURCE_1'], df[mask_target0]['EXT_SOURCE_1']]
data_source2 = [df[mask_target1]['EXT_SOURCE_2'], df[mask_target0]['EXT_SOURCE_2']]
data_source3 = [df[mask_target1]['EXT_SOURCE_3'], df[mask_target0]['EXT_SOURCE_3']]
group_labels = ['Défaillant', 'Non Défaillant']
colors = ['#333F44', '#37AA9C']


# Figure Source 1 
fig1 = ff.create_distplot(data_source1, group_labels, 
                         show_hist=False, 
                         colors=colors,
                         show_rug=False)

fig1.add_trace(go.Scatter(x=[np.array(df_id_client['EXT_SOURCE_1'])[0], np.array(df_id_client['EXT_SOURCE_1'])[0]], 
						  y=[-0.5, 2.5], mode="lines", name='Client', line=go.scatter.Line(color="red")))

fig1.update_layout(title={'text': "Source Extérieure 1",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, 
						 width=900,height=450,
						 xaxis_title="Source Ext 1",
    					 yaxis_title=" ",
    					 font=dict(size=15, color="#7f7f7f"))

fig1.update_yaxes(range=[-0.25, 2.4])

st.plotly_chart(fig1)

# Figure Source 2
fig2 = ff.create_distplot(data_source2, group_labels, 
                         show_hist=False, 
                         colors=colors,
                         show_rug=False)

fig2.add_trace(go.Scatter(x=[np.array(df_id_client['EXT_SOURCE_2'])[0], np.array(df_id_client['EXT_SOURCE_2'])[0]], y=[-1, 3.5], mode="lines", name='Client',
                         line=go.scatter.Line(color="red")))

fig2.update_layout(title={'text': "Source Extérieure 2",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, 
						 width=900,height=450,
						 xaxis_title="Source Ext 2",
    					 yaxis_title=" ",
    					 font=dict(size=15, color="#7f7f7f"))

fig2.update_yaxes(range=[-0.1, 3.1])

st.plotly_chart(fig2)

# Figure Source 3
fig3 = ff.create_distplot(data_source3, group_labels, 
                         show_hist=False, 
                         colors=colors,
                         show_rug=False)

fig3.add_trace(go.Scatter(x=[np.array(df_id_client['EXT_SOURCE_3'])[0], np.array(df_id_client['EXT_SOURCE_3'])[0]], y=[-0.5, 3.5], mode="lines", name='Client',
                         line=go.scatter.Line(color="red")))

fig3.update_layout(title={'text': "Source Extérieure 3",'xanchor': 'center', 'yanchor': 'top','y':0.9, 'x':0.5}, 
						 width=900,height=450,
						 xaxis_title="Source Ext 3",
    					 yaxis_title=" ",
    					 font=dict(size=15, color="#7f7f7f"))

fig3.update_xaxes(range=[0, 0.9])
fig3.update_yaxes(range=[-0.1, 2.9])


st.plotly_chart(fig3)