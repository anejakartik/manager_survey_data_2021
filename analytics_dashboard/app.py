import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from utility import plot_industry_boxplots_plotly, prepare_salary_experience_model_data, build_and_evaluate_random_forest_model,\
      get_region_from_state, convert_state_abbreviations, simplify_race
from utility import RegionSalaryAnalyzer, SalaryClassifier

# Load dataset
url = "cleaned_data.csv"  # Replace with actual data URL
data = pd.read_csv(url)
data['state'] = data['state'].apply(convert_state_abbreviations)
data['region'] = data['state'].apply(get_region_from_state)
data = data[data.state!='null']

# Create simplified race group column
data['race_grouped'] = data['race'].apply(simplify_race)
education_order = [
    'High School',
    'College degree',
    "Master's degree",
    'PhD',
]
data = data[data.race_grouped!="Not Specified"]

# Education level
data['education_level'].replace('Professional degree (MD, JD, etc.)', "Master's degree", inplace=True)
data['education_level'].replace('Some college', "College degree", inplace=True)
data['education_level'] = pd.Categorical(data['education_level'], categories=education_order, ordered=True)

analyzer = RegionSalaryAnalyzer(data)
metrics = analyzer.build_prediction_model()
print("Region Salary Analyzer : "+ str(metrics))


# Initialize SalaryClassifier
salary_classifier = SalaryClassifier()
X, y = salary_classifier.prepare_data(data)
results = salary_classifier.train_and_evaluate(X, y)
print("Salary Classifier : "+ str(results))

# Initialize the app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout
app.layout = html.Div([
    html.H1("Salary Analysis Dashboard", style={'textAlign': 'center'}),
    
    # Dropdown for analysis selection
    html.Label("Select Analysis:"),
    dcc.Dropdown(
        id='analysis-dropdown',
        options=[
            {'label': "Salary Distribution by Experience and Industry", 'value': 'salary_experience'},
            {'label': "Salary Levels by Industry and Location", 'value': 'salary_industry_location'},
            {'label': "Pay Disparity Across Demographics", 'value': 'pay_disparity'}
        ],
        value='salary_experience',  # Default selection
        clearable=False
    ),
    
    # Dynamic content based on dropdown
    html.Div(id='dynamic-content')
])
experience_order = [
    '1 year or less',
    '2 - 4 years',
    '5-7 years',
    '8 - 10 years',
    '11 - 20 years',
    '21 - 30 years',
    '31 - 40 years',
    '41 years or more'
]
# Callbacks for dynamic dashboard content
@app.callback(
    Output('dynamic-content', 'children'),
    [Input('analysis-dropdown', 'value')]
)
def update_dashboard(selected_analysis):
    if selected_analysis == 'salary_experience':
        # Analysis 1: Salary Distribution by Experience and Industry
        industries = data['major_industry'].unique()
        fig_box = plot_industry_boxplots_plotly(data, industries)
        
        # Median Salary by Industry
        median_salary_data = data.groupby('major_industry')['annual_salary'].median().reset_index()
        fig_median = px.bar(median_salary_data, x='major_industry', y='annual_salary', title="Median Salary by Industry")
        
        # Salary Experience Model
        X, y, industry_encoder, experience_encoder = prepare_salary_experience_model_data(data)
        model, metrics, _, _ = build_and_evaluate_random_forest_model(X, y)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        fig_model = px.bar(feature_importance, x='Feature', y='Importance', title="Salary Experience Model Feature Importance")
        
        return [
            html.H2("Salary Distribution by Experience and Industry"),
            dcc.Graph(figure=fig_box),
            html.H2("Median Salary by Industry"),
            dcc.Graph(figure=fig_median),
            html.H2("Feature Importance in Salary Prediction"),
            dcc.Graph(figure=fig_model)
        ]
    
    elif selected_analysis == 'salary_industry_location':
        # Analysis 2: Salary Levels by Industry and Location
        return [
            html.H2("State-wise Salary Distribution"),
            dcc.Graph(
                figure=px.box(
                    data,
                    x='region',
                    y='total_compensation',
                    color='state',
                    title="State-wise Salary Distribution",
                    labels={"region": "Region", "Salary": "Salary ($)", "state": "State"}
                )
            ),
            html.H2("Median Salaries by Industry and Region"),
            dcc.Graph(
                figure=px.density_heatmap(
                    data.groupby(['region', 'major_industry'])['total_compensation'].median().reset_index(),
                    x='region',
                    y='major_industry',
                    z='total_compensation',
                    color_continuous_scale='Viridis',
                    title="Median Salaries by Industry and Region",
                    width=1000,  # Increase width
                    height=800   # Increase height
                )
            ),
            html.H2("Region Salary Analysis"),
            html.Div([
                html.Label("Select Region"),
                dcc.Dropdown(
                    id='region-dropdown',
                    options=[{'label': reg, 'value': reg} for reg in data['region'].unique()],
                    value='Far West'  # Default region
                ),
                html.Div(id='region-salary-output')
            ])
        ]
    
    elif selected_analysis == 'pay_disparity':
        # Analysis 3: Pay Disparity Across Demographics
        # 1. Salary Distribution by Education Level
        fig_education = px.box(
            data,
            x='education_level',
            y='total_compensation',
            color='education_level',
            title="Salary Distribution by Education Level",
            labels={"total_compensation": "Salary ($)", "education_level": "Education Level"}
        )

        # 2. Salary Distribution by Gender
        fig_gender = px.box(
            data,
            x='gender',
            y='total_compensation',
            color='gender',
            title="Salary Distribution by Gender",
            labels={"total_compensation": "Salary ($)", "gender": "Gender"}
        )

        # 3. Salary Distribution by Racial Group
        fig_race = px.box(
            data,
            x='race_grouped',
            y='total_compensation',
            color='race_grouped',
            title="Salary Distribution by Racial Group",
            labels={"total_compensation": "Salary ($)", "race_grouped": "Racial Group"}
        )

        # 4. Correlation Heatmap: Salary vs Key Factors
        encoded_df = pd.get_dummies(data, columns=['gender', 'race_grouped', 'education_level', 'major_industry', 'overall_experience'])

        fig_corr = ff.create_annotated_heatmap(
            z=encoded_df[['total_compensation', 'gender_Man', 'gender_Woman', 'education_level_PhD', 
    'education_level_Master\'s degree']].corr().values,
            x=encoded_df[['total_compensation', 'gender_Man', 'gender_Woman', 'education_level_PhD', 
    'education_level_Master\'s degree']].corr().columns.tolist(),
            y=encoded_df[['total_compensation', 'gender_Man', 'gender_Woman', 'education_level_PhD', 
    'education_level_Master\'s degree']].corr().columns.tolist(),
            colorscale='Viridis'
        )
        fig_corr.update_layout(
            title="Correlation Heatmap: Salary vs Key Factors"
        )
        print(data['education_level'].unique())
        # 5. Salary Classification Dropdowns and Prediction
        return [
            html.H2("Salary Distribution by Education Level"),
            dcc.Graph(figure=fig_education),
            
            html.H2("Salary Distribution by Gender"),
            dcc.Graph(figure=fig_gender),
            
            html.H2("Salary Distribution by Racial Group"),
            dcc.Graph(figure=fig_race),
            
            html.H2("Correlation Heatmap: Salary vs Key Factors"),
            dcc.Graph(figure=fig_corr),
            
            html.H3("Predict Salary Level"),
            html.Label("Select Gender"),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[{'label': gender, 'value': gender} for gender in data['gender'].unique()],
                value=data['gender'].unique()[0]
            ),
            html.Label("Select Education Level"),
            dcc.Dropdown(
                id='education-dropdown',
                options=[{'label': edu, 'value': edu} for edu in data['education_level'].unique() if not (pd.isna(edu) or pd.isnull(edu)) ],
                value=data['education_level'].unique()[0]
            ),
            html.Label("Select Racial Group"),
            dcc.Dropdown(
                id='race-dropdown',
                options=[{'label': race, 'value': race} for race in data['race_grouped'].unique()],
                value=data['race_grouped'].unique()[0]
            ),
            html.Label("Select Industry"),
            dcc.Dropdown(
                id='industry-dropdown',
                options=[{'label': industry, 'value': industry} for industry in data['major_industry'].unique()],
                value=data['major_industry'].unique()[0]
            ),
            html.Label("Select Experience Level"),
            dcc.Dropdown(
                id='experience-dropdown',
                options=[{'label': exp, 'value': exp} for exp in experience_order],
                value=experience_order[0]
            ),
            html.Button("Predict Salary Level", id='predict-button'),
            html.Div(id='classification-output')
        ]

# Callback for RegionSalaryAnalyzer
@app.callback(
    Output('region-salary-output', 'children'),
    [Input('region-dropdown', 'value')]
)
def update_region_salary_analysis(selected_region):
    region_comparison = analyzer.compare_industries(selected_region)
    region_comparison_df = pd.DataFrame(region_comparison.items(), columns=['Industry', 'Predicted Salary'])
    fig_region_analysis = px.bar(
        region_comparison_df,
        x='Industry',
        y='Predicted Salary',
        title=f"Predicted Salaries for Industries in {selected_region}"
    )
    return dcc.Graph(figure=fig_region_analysis)

# Callback for Salary Prediction
@app.callback(
    Output('classification-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [Input('gender-dropdown', 'value'),
     Input('education-dropdown', 'value'),
     Input('race-dropdown', 'value'),
     Input('industry-dropdown', 'value'),
     Input('experience-dropdown', 'value')]
)
def predict_salary(n_clicks, gender, education, race, industry, experience):
    if n_clicks:
        # Predict salary category
        salary_category = salary_classifier.predict_salary_category(
            gender=gender,
            race=race,
            education=education,
            industry=industry,
            experience=experience
        )
        return html.Div([
            html.P(f"Predicted Salary Category: {salary_category}")
        ])
    return html.Div("Click 'Predict Salary Level' to see the results.")

if __name__ == '__main__':
    app.run_server(debug=True)
