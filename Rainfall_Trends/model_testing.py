import pickle
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
print(type(model))

# create a future dataframe for the next 20 years
future = model.make_future_dataframe(periods=20, freq='YE')
forecast = model.predict(future)

fig_forecast = plot_plotly(model, forecast)

fig_forecast.update_layout(
    title='Annual Rainfall Forecast Using Prophet',
    xaxis_title='Year',
    yaxis_title='Rainfall (mm)',
    template='plotly_white',
    height=500
)

fig_forecast.show()
