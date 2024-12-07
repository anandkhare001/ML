import pickle
import streamlit as st
from prophet.plot import plot_plotly, plot_components_plotly


# App Title
st.title('Rainfall Forecast')

# Get user input for number of years for forecast
years = st.number_input('Number of years for forecast', min_value=1, max_value=50, value=1)

# Simulate historical rainfall data (for demo purposes)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# create a future dataframe for the next 20 years
future = model.make_future_dataframe(periods=years, freq='YE')
forecast = model.predict(future)

fig_forecast = plot_plotly(model, forecast)

fig_forecast.update_layout(
    title=f'Rainfall Forecast for the Next {years} Year(s)',
    xaxis_title='Year',
    yaxis_title='Rainfall (mm)',
    template='plotly_white',
    height=500
)


# Plot the forecast
st.plotly_chart(fig_forecast)

# Show the forecasted data in a table format
st.write(f'Forecasted rainfall for the next {years} year(s):')
forecasted_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
st.write(forecasted_data.tail(12))  # Show the forecast for the last 12 months
