# Python version.
FROM python:3.9

# Expose a port.
EXPOSE 8080

# Install requirements.
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . ./

# Add comments.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]




