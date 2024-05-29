import gradio as gr
import pandas as pd
import numpy as np
import joblib
from scipy.stats import boxcox

numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
target_label = ['NObeyesdad']

def load_model(model):
    if model == "Logistic regression":
        pred_model = joblib.load("checkpoint/logistic_regression.joblib")
    elif model == "K nearest neighbors":
        pred_model = joblib.load("checkpoint/knn.joblib")
    elif model == "Decision tree":
        pred_model = joblib.load("checkpoint/decision_tree.joblib")
    elif model == "Random forest":
        pred_model = joblib.load("checkpoint/random_forest.joblib")
    elif model == "XGBoost":
        pred_model = joblib.load("checkpoint/xg_boost.joblib")
    elif model == "Voting classifier":
        pred_model = joblib.load("checkpoint/votingClassifier.joblib")
    return pred_model

preprocessing = joblib.load("checkpoint/preprocessing.joblib")

label_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

def predict_obesity_level(model_name, age, height, weight, fcvc, ncp, ch2o, faf, tue, 
                          gender, family_history_with_overweight, favc, caec, smoke, 
                          scc, calc, mtrans):
    model = load_model(model_name)
    
    x = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })
    
    if age > 0:
        try:
            x['Age'], _ = boxcox(x['Age'])
        except ValueError:
            x['Age'] = np.log1p(x['Age'])
    else:
        x['Age'] = np.log1p(x['Age'])
    
    x['FCVC'] = pd.cut(x['FCVC'], bins=[0.5,1.5,2.5,3.5], labels=[1,2,3]).astype('float64')
    x['NCP'] = pd.cut(x['NCP'], bins=[0.5,1.5,2.5,3.5,4.5], labels=[1,2,3,4]).astype('float64')
    x['CH2O'] = pd.cut(x['CH2O'], bins=[0.5,1.5,2.5,3.5], labels=[1,2,3]).astype('float64')
    x['FAF'] = pd.cut(x['FAF'], bins=[-0.5,0.5,1.5,2.5,3.5], labels=[0,1,2,3]).astype('float64')
    x['TUE'] = pd.cut(x['TUE'], bins=[-0.5,0.5,1.5,2.5], labels=[0,1,2]).astype('float64')
    
    int64_columns = x.select_dtypes(include='int64').columns
    x[int64_columns] = x[int64_columns].astype('float64')
    
    x = preprocessing.transform(x)
    x = pd.DataFrame(x, columns=preprocessing.get_feature_names_out())
    y = model.predict(x)
    return label_mapping[y[0]]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
    gr.Markdown("# Obesity Level Classification")
    gr.Markdown("Predict the level of obesity based on various health and lifestyle factors.")
    gr.Markdown("Note: ")
    gr.Markdown("The value of Consumption of vegetables (FCVC) ranges from 1 to 3.")
    gr.Markdown("The value of Number of main meals (NCP) ranges from 1 to 4.")
    gr.Markdown("The value of Consumption of water daily (CH2O) ranges from 1 to 3.")
    gr.Markdown("The value of Physical activity frequency (FAF) ranges from 0 to 3.")
    gr.Markdown("The value of Time using tech devices (TUE) ranges from 0 to 2.")

    with gr.Group():
        gr.Markdown("## Personal Status")
        with gr.Row():
            Age = gr.Number(label="Age")
            Height = gr.Number(label="Height")
            Weight = gr.Number(label="Weight")
        with gr.Row():
            Gender = gr.Dropdown(label="Gender", choices=["Male", "Female"])
            Family_history = gr.Dropdown(label="Family history with overweight", choices=["yes", "no"])

    with gr.Group():
        gr.Markdown("## Routine")
        with gr.Row():
            FAF = gr.Number(label="Physical activity frequency (FAF)", minimum=0, maximum=3, step=1)
            TUE = gr.Number(label="Time using tech devices (TUE)", minimum=0, maximum=2, step=1)
        with gr.Row():
            MTRANS = gr.Dropdown(label="Mode of transportation (MTRANS)", choices=["Automobile", "Motorbike", "Bike", "Public Transportation", "Walking"])
    
    with gr.Group():
        gr.Markdown("## Eating Habits")
        with gr.Row():
            FCVC = gr.Number(label="Consumption of vegetables (FCVC)", minimum=1, maximum=3, step=1)
            NCP = gr.Number(label="Number of main meals (NCP)", minimum=1, maximum=4, step=1)
        with gr.Row():
            FAVC = gr.Dropdown(label="Consumption of high caloric food (FAVC)", choices=["yes", "no"])
            CAEC = gr.Dropdown(label="Consumption of food between meals (CAEC)", choices=["no", "Sometimes", "Frequently", "Always"])
            CH2O = gr.Number(label="Consumption of water daily (CH2O)", minimum=1, maximum=3, step=1)

    with gr.Group():
        gr.Markdown("## Health-related Factors")
        with gr.Row():
            SMOKE = gr.Dropdown(label="Smokes (SMOKE)", choices=["yes", "no"])
            SCC = gr.Dropdown(label="Monitor calories consumption (SCC)", choices=["yes", "no"])
        with gr.Row():
            CALC = gr.Dropdown(label="Consumption of alcohol (CALC)", choices=["no", "Sometimes", "Frequently", "Always"])
        
    Model = gr.Dropdown(
        label="Model",
        choices=[
            "Logistic regression",
            "K nearest neighbors",
            "Decision tree",
            "Random forest",
            "XGBoost",
            "Voting classifier"
        ]
    )

    Prediction = gr.Textbox(label="Obesity Level Classification")

    with gr.Row():
        submit_button = gr.Button("Predict")
        submit_button.click(fn=predict_obesity_level,
                            outputs=Prediction,
                            inputs=[Model, Age, Height, Weight, FCVC, NCP, CH2O, FAF, TUE,
                                    Gender, Family_history, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS
                                    ],
                            queue=True)
        clear_button = gr.ClearButton(components=[Prediction], value="Clear")
        
    app.launch()