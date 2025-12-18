import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
 
#Page Configuration
st.set_page_config("Linear Regression", layout="centered")
 
# Loading CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
load_css("styles.css")
 
# Title
st.markdown("""<html>
    <body>
    <div class="card">
        <h1>Linear Regression Model</h1>
        <p> Predict <b> Tip Amount </b> from <b> Total Bill </b> using Linear Regression</p>
    </div>
    </body>
</html>""", unsafe_allow_html=True)
 
# Load Dataset
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df = load_data()
 
# Dataset Preview
 
st.markdown('<div class="card"><h3>Dataset Preview</h3>', unsafe_allow_html=True)
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)
 
# Prepare Data
x=df[['total_bill','size']]
y=df['tip']
 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
 
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
 
#Train Model
model=LinearRegression()
model.fit(x_train_scaled,y_train)
 
y_pred=model.predict(x_test_scaled)
 
#Metrics
 
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)
 
# Visualization
 
st.markdown('<div class="card"><h3>Total Bill vs Tip</h3>', unsafe_allow_html=True)
fig, ax = plt.subplots()
ax.scatter(df['total_bill'], df['tip'], color='#7f1d1d', alpha=0.6)
ax.plot(df["total_bill"], model.predict(scaler.transform(df[['total_bill','size']])), color="#d15858", linewidth=2)
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
 
#Performance Metrics
st.markdown('<div class="card"><h3>Model Performance Metrics</h3>', unsafe_allow_html=True)
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c3.metric("R²",f"{r2:.2f}")
c4.metric("Adjusted R²",f"{adj_r2:.2f}")
st.markdown('</div>', unsafe_allow_html=True)
 
# m1,m2 & c
m1=model.coef_[0]
m2=model.coef_[1]
c=model.intercept_
 
st.markdown(f"""
    <div class="card">
        <h3>Model Interception</h3>
        <p><b> Co-efficient: </b>{m1:.3f}<br>
        <b>Co-efficient: </b>{m2:.3f}<br>
        <b>Intercept: </b>{c:.3f}</p>
    </div>
""",unsafe_allow_html=True
)
 
# Prediction
min_bill,max_bill=float(df.total_bill.min()), float(df.total_bill.max())
bill = st.slider("Enter Total Bill",
    min_value=min_bill,
    max_value=max_bill,
    value=(min_bill + max_bill) / 2
)

min_size,max_size = int(df['size'].min()), int(df['size'].max())
size = st.slider("Enter Size",
    min_value=min_size,
    max_value=max_size,
    value=(min_size + max_size) // 2
)
tip=model.predict(scaler.transform([[bill,size]]))[0]
st.markdown(f'<div class="prediction-box"><h3>Predicted Tip for Total Bill ${bill:.2f} is <b>${tip:.2f}</b></h3></div>', unsafe_allow_html=True)
