import numpy as np
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


@st.cache
def get_data():
    return pd.read_csv("pizza.csv")


@st.cache(allow_output_mutation=True)
def get_model():
    return pickle.load(open("linear.model", "rb"))


lr = get_model()

promote = st.sidebar.slider("Promotion", 0, 100, 50)

y_pred = lr.predict([[promote]])
y_pred = int(y_pred.round())
st.title(f"Sales: {y_pred}")
st.subheader(f"y={lr.intercept_.round(2)}+x({lr.coef_[0].round(2)})")
df = get_data()

X = df['Promote'].values[:, np.newaxis]
y = df['Sales']

fig, ax = plt.subplots()
df = pd.get_dummies(df)
ax.scatter(X, y, c="#ADCEFE")
ax.scatter([[promote]], lr.predict([[promote]]), s=300, c="#4E9F71")
ax.plot(X, lr.predict(X))
ax.annotate(f"Predicted Sales \n {promote}, {y_pred}", (promote, y_pred))
plt.xlabel("Promote")
plt.ylabel("Sales")
st.pyplot(fig)
