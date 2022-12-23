## Title: Income Prediction
    The prominent inequality of wealth and income is a huge concern especially in the United States. The likelihood of diminishing poverty is one valid reason to reduce the world's surging level of economic inequality. The principle of universal moral equality ensures sustainable development and improve the economic stability which governments’ try their best to address this issue. The problem of income inequality has been of great concern in the recent years. Making the poor better off does not seem to be the sole criteria to be in quest for eradicating this issue. People of the United States believe that the advent of economic inequality is unacceptable and demands a fair share of wealth in the society. The fields of Data Mining and Machine Learning have not only exploited the available data for knowledge and discovery but also to explore certain hidden patterns and concepts which led to the prediction of future events

## Problem Motivation:
    Aims to conduct a comprehensive analysis to highlight the key factors that are necessary in improving an individual's income. Set focus on the important areas which can significantly improve the income levels of individuals. Classification has been done to predict whether a person's yearly income in US falls in the income category of either greater than 50K Dollars or less equal to 50K Dollar’s category based on a certain set of attributes. Simplify the data system by reducing the number of variables to be studied, without sacrificing too much of accuracy. 

## Getting Started
Dataset: https://www.kaggle.com/datasets/mastmustu/income

### Dependencies
Describing the prerequisites, libraries, OS version, etc., needed for our application.
    Operating System: Windows 10
    IDE : Visual Studio Code
    Python Version: Python 3.10.7
    Package Manager: pip
    Framework : Streamlit
    Packages:Pandas, Seaborn, Matplotlib, Numpy, Sklearn, Pickle

### Installing
1. How/where to download your program
    https://www.python.org/downloads/ - Python Download
    https://code.visualstudio.com/download - Visual Studio Code Download

2. Installing Streamlit:
    1. Open terminal in VSCode 
    2. When the terminal appears type "pip install streamlit"
    3. To test the installation type "Streamlit hello"

3. Installing Packages:
    1. pip install pandas
    2. pip install matplotlib
    3. pip install seaborn
    4. pip install sklearn
    5. pip install numpy
    6. pip install pickle

### Executing program
Step 1:Open folder named induraju_kaviyaav_phase3.
    app.py - The root file to run the web application. 
    explore_model.py - This file contains visualization code
    predict_education.py - This file contains front end code for education prediction 
    predict_income.py - This file contains front end code income prediction 
    phase3_rf_education.ipynb - This file contains back end code for education prediction
    phase3_rf.ipynb - This file contains back end code for income prediction
    random_forest_degree_pred.pkl - Pickle file for education prediction used in predict_education.py
    random_forest_income_pred.pkl - Pickle file for income prediction used in predict_income.py
    income.csv - Data source
Step 2:Run the command "streanlit run app.py" in the terminal.
Step 3:User Interface will be rendered in the localhost:8501


## Help
For downloading Streamlit in Mac or Linux
    1. Install pip
    On a macOS:
    python -m ensurepip --upgrade

    On Ubuntu with Python 3:
    sudo apt-get install python3-pip

    2. Install pipenv.
    pip3 install pipenv

    3. Install Xcode command line tools on macOS
    On macOS, you'll need to install Xcode command line tools. They are required to compile some of Streamlit's Python dependencies during installation. To install Xcode command line tools, run:
    "xcode-select --install"

Create a new environment with Streamlit
    1. Navigate to your project folder:
    cd myproject

    2. Create a new Pipenv environment in that folder and activate that environment:
    pipenv shell
    When you run the command above, a file called Pipfile will appear in myprojects/. This file is where your Pipenv environment and its dependencies are declared.

    3. Install Streamlit in your environment:
    pip install streamlit
    Or if you want to create an easily-reproducible environment, replace pip with pipenv every time you install something:
    pipenv install streamlit

    4. Test that the installation worked:
    streamlit hello

## Authors
Contributors names and contact info:
Indu Raju - induraju@buffalo.edu
Kaviyaa Vasudevan - kaviyaav@buffalo.edu