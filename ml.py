import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("🌡️ Temperature Prediction")
st.sidebar.header("Enter Weather Details")
humidity = st.sidebar.slider("Humidity (%)", 0, 100)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50)
previous_temp = st.sidebar.number_input("Previous Day Temp (°C)", min_value=0.0, max_value=50.0)
if st.sidebar.button("sumbit"):
	st.write("Input Summary")
	st.write(f"Humidity: {humidity}%")
	st.write(f"Wind Speed: {wind_speed} km/h")
	st.write(f"Previous Temperature: {previous_temp} °C")
df=pd.read_csv("temp_data.csv")
x=df[["Humidity","Wind_Speed","Previous_Temp"]]
y=df["Today_Temp"]
p=LinearRegression()
p.fit(x,y)
predicted_temp=p.predict([[humidity,wind_speed,previous_temp]])
if st.button("Predict"):
	st.write(f"Predicted Today Temperature:{predicted_temp} °C")
st.header("📊 View Historical Data Graph?")
if st.button("Yes"):
    st.subheader("Temperature vs Humidity and Wind Speed")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df["Humidity"], df["Wind_Speed"], c=df["Today_Temp"], cmap='coolwarm', s=100)
    ax.set_xlabel("Humidity (%)")
    ax.set_ylabel("Wind Speed (km/h)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Today Temp (°C)")
    st.pyplot(fig)
result_df = pd.DataFrame({
    "Humidity": [humidity],
    "Wind Speed": [wind_speed],
    "Previous Temp": [previous_temp],
    "Predicted Temp": [predicted_temp[0]]
})
csv = result_df.to_csv(index=False)
st.download_button("📥 Download Prediction as CSV", data=csv, file_name="prediction.csv", mime="text/csv")



