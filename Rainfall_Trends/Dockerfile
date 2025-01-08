# Python version
FROM python:3.9

# Set a working directory
WORKDIR /usr/src/rainfall-trends

# Copy all the files to the container
COPY . .

# Install Dependencies
RUN pip install --upgrade pip --progress-bar off
RUN pip install --no-cache-dir -r requirements.txt --progress-bar off

# Set port to expose app
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "streamlit_rainfall_forecast_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
