import streamlit as st
import numpy as np
import pickle

df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))

st.title('Laptop Prediction')
company = st.selectbox('Brand',df['Company'].unique())
type = st.selectbox('Type',df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
wt = st.number_input('Weight')
ts = st.selectbox('Touchscreen',['No','Yes'])
scrsiz = st.number_input('Screen Size')
reso = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox('CPU',df['CPU Brand'].unique())
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
gpu = st.selectbox('GPU',df['GPU Brand'].unique())
os = st.selectbox('OS',df['OS Name'].unique())

if st.button('Predict Price'):
    ppi = None
    if ts == 'Yes':
        ts = 1
    else:
        ts = 0
    X_res = int(reso.split('x')[0])
    Y_res = int(reso.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/scrsiz
    query = np.array([company,type,ram,wt,ts,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,11)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
