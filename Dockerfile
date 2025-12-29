# Use Python 3.11 to support the latest libraries like 'contourpy'
FROM python:3.11

# Set working directory
WORKDIR /code

# Copy requirements 
COPY ./requirements.txt /code/requirements.txt

# Install dependencies (Opencv support)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Install Python libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files
COPY . .

# Grant permission to execute the script
RUN chmod +x run.sh

# Run the shell script
CMD ["./run.sh"]