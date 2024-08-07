# AgriSage- The Sage of Agriculture
This project focuses on the development and implementation of a predictive model for crop selection based on a comprehensive agricultural dataset. The dataset comprises multiple features such as nutrients in soil, weather conditions, and other agronomic factors, with the target variable being the crop type.

The primary objective was to analyze the dataset to understand the key factors influencing crop selection, followed by the construction of a predictive model to recommend the most suitable crops for given conditions. Various data preprocessing techniques and exploratory data analysis (EDA) were employed to enhance the quality of the data.

Multiple machine learning algorithms were evaluated, including decision trees, random forests, and logistic regression, to determine the most accurate model. The final model was selected based on its accuracy.

Additionally, a Streamlit application was developed to provide an interactive user interface for stakeholders, enabling them to input specific agronomic conditions and receive crop recommendations with associated probabilities. This tool aims to support farmers and agricultural planners in making informed decisions to optimize crop yields and sustainability. Additionally, it ensures the validity of the model presented.

## Project Report
The project report can be viewed [here](https://subhangi03.github.io/AgriSage/report/report.html).

## Streamlit App
Here is the working of the final Streamlit App explained via screenshots:
### First look of the app:
![First look of the app](Streamlit_screenshots/Screenshotfirst.png)
### App options:
![App options](Streamlit_screenshots/Screenshotsecond.png)
### User input (7 such features):
![User input (7 such features)](Streamlit_screenshots/Screenshotthird.png)
### Final user input:
![Final user input](Streamlit_screenshots/Screenshotfourth.png)
### Prediction results:
![Results](Streamlit_screenshots/Screenshotappresults5.png)
### Option to validate the model:
![Option to validate the model](Streamlit_screenshots/Screenshotsix.png)
### Model validation results:
![Model validation results](Streamlit_screenshots/Screenshotmodelvalidationseven.png)

## Steps used to set up a virtual environment:
```
python3.12.3 -m venv agrisage.venv

pip install -r requirements.txt

```
I have used libraries like pandas, seaborn and matplot. For further information on the libraries used, check 'requirements.txt' file. 