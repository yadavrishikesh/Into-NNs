rm(list = ls())
library(keras)
library(tensorflow)
library(reticulate)

version <- "3.12.4"
# Create a virtual environment 'myenv' with Python 3.12.4 and install TensorFlow within this environment
reticulate::virtualenv_create(envname = 'myenv',
                              python = "/usr/local/bin/python3",  # Path to Python 3.12.4
                              version = version)

# Set the Python interpreter to the one in the created virtual environment
path <- paste0(reticulate::virtualenv_root(), "/myenv/bin/python3")
Sys.setenv(RETICULATE_PYTHON = "/myenv/bin/python3")

reticulate::use_virtualenv("myenv", required = TRUE)
reticulate::virtualenv_install("myenv", packages = "tensorflow")  # Install TensorFlow in the virtual environment

# Install Keras using the virtual environment
install_keras(method = "virtualenv", envname = "myenv")

# Check if Keras is available
is_keras_available()



library(keras)
library(tensorflow)
library(reticulate)

# Set Python interpreter to that installed in the virtual environment
Sys.setenv(RETICULATE_PYTHON = "/Users/rishikeshyadav/.virtualenvs/myenv/bin/python3")

# Check Python configuration
reticulate::py_config()

# Check if TensorFlow is available
tensorflow::tf_config()

# Install Keras in the specified environment
install_keras(method = "virtualenv", envname = "myenv")

# Check if Keras is available
is_keras_available()
