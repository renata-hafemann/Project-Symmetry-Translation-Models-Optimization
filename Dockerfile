# Use the slim version of Python 3.10.2 as the base image
FROM python:3.10.2-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /webproject

# Copy the entire project directory into the container
COPY . /webproject/

# Copy the requirements file into the container
COPY ./requirements.txt /webproject/

# Install required Python packages
RUN pip install --no-cache-dir -r /webproject/requirements.txt

# Expose port 8000
EXPOSE 8000

# Set the command to run when the container starts
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]