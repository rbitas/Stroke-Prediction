from dash import Dash, html, dcc, callback, Output, Input, State
import pickle
import pandas as pd

external_stylesheets = [
    'https://unpkg.com/simpledotcss/simple.min.css',
]

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)
with open("classifier.pkl","rb") as f:
    model = pickle.load(f)

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children='Stroke Prediction'),
    html.H3(children='Group 5 Project 4: Machine Learning'),
    html.P(children='Enter your information to see if you are likely to get a stroke'),
    html.Br(),
    dcc.Dropdown(
        [0,1,2],
        id='dropdown-malefemale',
        placeholder="0 if you are Male, 1 if you are Female, or 2 if Other",
        style= {
            "width": "30em"
        },
    ),
    html.Br(),
    dcc.Input(
        id='input-age',
        type='number',
        placeholder="Enter your age",
        style= {
            "width": "50em"
        },
    ),
    html.Br(),
    dcc.Dropdown(
        [0,1],
        id='dropdown-hypertension',
        placeholder="Has Hypertension [0] or doesn't have Hypertension [1]",
        style= {
            "width": "50em"
        },
    ),
    html.Br(),
    dcc.Dropdown(
        [0,1],
        id='dropdown-heartdisease',
        placeholder="Has Heart disease [0] or doesn't have Heart disease [1]",
        style= {
            "width": "50em"
        },
    ),
    html.Br(),
    dcc.Input(
        id='input-avgglucoselvl',
        type='number',
        placeholder="Enter your average glucose level",
        style= {
            "width": "75em"
        },
    ),
    html.Br(),
    dcc.Input(
        id='input-bmi',
        type='number',
        placeholder="Enter your bmi",
        style= {
            "width": "75em"
        },
    ),
    html.Br(),
    dcc.Dropdown(
        [0,1],
        id='dropdown-evermarried',
        placeholder="0 if you were never married, 1 if you were ever married",
        style= {
            "width": "50em"
        },
    ),
    html.Br(),
    dcc.Dropdown(
        [0,1,2,3,4],
        id='dropdown-worktype',
        placeholder="0 if you never worked, 1 if you work private, 2 if you are self employed, 3 if you work a government job, 4 if you are a child",
        style= {
            "width": "50em"
        },
    ),    
    html.Br(),
    dcc.Dropdown(
        [0,1],
        id='dropdown-residencetype',
        placeholder="0 if Rural, 1 if Urban",
        style= {
            "width": "50em"
        },
    ),
    html.Br(),
    dcc.Dropdown(
        [0,1,2,3],
        id='dropdown-smokingstatus',
        placeholder="Prefer not to say [0], Formerly smoked [1], Never smoked[2], Smokes currently[3]",
        style= {
            "width": "50em"
        },
    ),
    html.Br(),
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Br(),
    html.Br(),
    html.H3(children="Results"),
    html.P(children="", id="p-result")
])

@callback(
    Output('p-result', 'children'),
    Input('submit-val', "n_clicks"),
    State('dropdown-malefemale', 'value'),
    State('input-age', 'value'),
    State('dropdown-hypertension', 'value'),
    State('dropdown-heartdisease', 'value'),
    State('input-avgglucoselvl', 'value'),
    State('input-bmi', 'value'),
    State('dropdown-evermarried', 'value'),
    State('dropdown-worktype', 'value'),    
    State('dropdown-residencetype', 'value'),
    State('dropdown-smokingstatus', 'value'),
    prevent_initial_call=True
)
def update_result(clicks, gender, age, 
                  hypertension, heartdisease, avgglucoselvl, bmi,evermarried, worktype, residencetype,smokingstatus):
    info_for_prediction = {
        "gender_Female": 1 if gender== 1 else 0,
        "gender_Male": 1 if gender== 0 else 0,
        "gender_other": 1 if gender== 2 else 0, 
        "age": float(age),
        "hypertension": float(hypertension),
        "heart_disease": float(heartdisease),
        "avgglucoselevel": float(avgglucoselvl),
        "bmi": float(bmi),
        "ever_married_No": 1 if evermarried == 0 else 0,
        "ever_married_Yes": 1 if evermarried == 1 else 0,
        "work_type_Govt_job": 1 if worktype == 3 else 0,
        "work_type_Never_worked": 1 if worktype == 0 else 0,
        "work_type_Private": 1 if worktype == 1 else 0,
        "work_type_Self-employed": 1 if worktype == 2 else 0,
        "work_type_children": 1 if worktype == 4 else 0,    
        "Residence_type_Rural": 1 if residencetype == 0 else 0,
        "Residence_type_Urban": 1 if residencetype == 1 else 0, 
        "smoking_status_Unknown": 1 if smokingstatus == 0 else 0,
        "smoking_status_formerly smoked": 1 if smokingstatus == 1 else 0,
        "smoking_status_never smoked": 1 if smokingstatus == 2 else 0,
        "smoking_status_smokes": 1 if smokingstatus == 3 else 0,
        }
    df_predict = pd.DataFrame(info_for_prediction,index=[0])
    df_predict = scaler.transform(df_predict)
    answer = model.predict(df_predict)
    if answer == 0:
        result = "You are less likely to get a stroke. Congratulations!"
    else:
        result = "You are more likely to get a stroke. Please seek out the attention of a medical professional."
    return result

if __name__ == '__main__':
    app.run(debug=True)