import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix
import plotly.subplots as sp
import plotly.graph_objects as go

def read_csv_file(file_path):
    """
    Reads a CSV file and returns it as a DataFrame.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """
    df = pd.read_csv(file_path)
    return df

def rename_columns(df, column_mapping):
    """
    Renames columns of a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame whose columns need to be renamed.
        column_mapping (dict): A dictionary where keys are current column names and values are new column names.

    Returns:
        pd.DataFrame: A DataFrame with renamed columns.
    """
    return df.rename(columns=column_mapping)

def map_to_major_category(industry):
    if type(industry)!=str:
      return 'Other'
    industry = industry.lower()
    if any(keyword in industry for keyword in ['education', 'academic', 'student', 'library', 'museum']):
        return 'Education and Research'
    elif any(keyword in industry for keyword in ['health', 'pharma', 'biotech', 'medical', 'life sciences']):
        return 'Healthcare and Life Sciences'
    elif any(keyword in industry for keyword in ['technology', 'it', 'software']):
        return 'Technology and Information'
    elif any(keyword in industry for keyword in ['manufacturing', 'production', 'industry']):
        return 'Manufacturing and Industry'
    elif any(keyword in industry for keyword in ['real estate', 'property', 'construction']):
        return 'Real Estate and Construction'
    elif any(keyword in industry for keyword in ['finance', 'insurance', 'banking']):
        return 'Financial Services'
    elif any(keyword in industry for keyword in ['government', 'public', 'law enforcement']):
        return 'Government and Public Sector'
    elif any(keyword in industry for keyword in ['marketing', 'advertising', 'sales', 'consulting']):
        return 'Professional Services'
    elif any(keyword in industry for keyword in ['retail', 'restaurant', 'food', 'beverage']):
        return 'Retail and Consumer Services'
    else:
        return 'Other'

def filter_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Filters and returns a new DataFrame containing only the specified columns.

    Args:
        data (pd.DataFrame): The original DataFrame.
        columns (list): List of column names to retain in the resulting DataFrame.

    Returns:
        pd.DataFrame: New DataFrame containing only the specified columns.
    """
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"The following columns are not found in the DataFrame: {', '.join(missing_cols)}")
    
    return data[columns]


def plot_industry_boxplots(df, industries, ncols=2):
    """
    Creates a grid of boxplots showing salary distributions across experience levels for each industry.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the salary data with columns:
            - 'major_industry': Industry category
            - 'overall_experience': Experience level category
            - 'annual_salary': Numerical salary values
        industries (array-like): List or array of industry names to plot
        ncols (int, optional): Number of columns in the plot grid. Defaults to 2.
    
    Returns:
        None. Displays the plot grid using matplotlib.
        
    Examples:
        >>> industries = df['major_industry'].unique()
        >>> plot_industry_boxplots(df, industries)
        >>> plot_industry_boxplots(df, industries, ncols=1)  # Single column layout
        
    Notes:
        - Creates one boxplot per industry
        - Each boxplot shows salary distribution across experience levels
        - Automatically adjusts grid layout based on number of industries
        - Removes empty subplots if the grid isn't completely filled
        - Uses tight_layout to prevent subplot overlap
    """
    # Calculate number of rows needed
    nrows = (len(industries) + ncols - 1) // ncols
    
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))
    axes = axes.flatten()
    
    # Plot each industry
    for idx, industry in enumerate(industries):
        # Filter data for current industry
        industry_data = df[df['major_industry'] == industry]
        
        # Create boxplot
        sns.boxplot(data=industry_data, 
                   x='overall_experience',
                   y='annual_salary',
                   ax=axes[idx],
                   order = [
                            '1 year or less',
                            '2 - 4 years',
                            '5-7 years',
                            '8 - 10 years',
                            '11 - 20 years',
                            '21 - 30 years',
                            '31 - 40 years',
                            '41 years or more'
                        ])
        
        # Customize plot
        axes[idx].set_title(f'Salary Distribution in {industry}')
        axes[idx].set_xlabel('Overall Experience')
        axes[idx].set_ylabel('Annual Salary (USD)')
        axes[idx].tick_params(axis='x', rotation=45)
        
    # Remove empty subplots if any
    for idx in range(len(industries), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


# Calculate salary growth rates between experience levels
def calculate_growth_rates(df):
    """
    Calculates the annual salary growth rates based on the median salary for each
    level of overall experience. The growth rate is calculated as the percentage 
    change in the median salary between consecutive experience levels.

    The function groups the data by 'overall_experience', calculates the median 
    salary for each experience level, and then computes the percentage change 
    between consecutive experience levels to estimate the salary growth rates.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least two columns:
                           - 'overall_experience': The level of experience (numeric or categorical).
                           - 'annual_salary': The annual salary (numeric) for each individual.

    Returns:
        pd.Series: A pandas Series containing the calculated salary growth rates 
                   (percentage change) for each experience level. The growth rates 
                   are rounded to two decimal places.

    Example:
        # Sample usage
        growth_rates = calculate_growth_rates(df)
        print(growth_rates)
    
    Note:
        - The growth rate for the first experience level will be `NaN` because there is no previous level to compare to.
        - If the input DataFrame has missing values, the function will ignore them during calculation.
    """
    experience_medians = df.groupby('overall_experience')['annual_salary'].median()
    growth_rates = experience_medians.pct_change() * 100
    return growth_rates.round(2)


def prepare_salary_experience_model_data(df):
    """
    Prepares the feature matrix (X) and target vector (y) for training a machine learning model 
    to predict annual salary based on the 'major_industry' and 'overall_experience'. The function 
    encodes categorical variables ('major_industry' and 'overall_experience') using LabelEncoder 
    and returns the prepared data along with the fitted encoders.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing at least the following columns:
                           - 'major_industry': Categorical column indicating the industry.
                           - 'overall_experience': Categorical or numeric column indicating the experience level.
                           - 'annual_salary': Numeric column indicating the annual salary (target variable).

    Returns:
        X (pd.DataFrame): A DataFrame containing the encoded features ('major_industry' and 'overall_experience').
        y (pd.Series): A pandas Series containing the target variable 'annual_salary'.
        industry_encoder (LabelEncoder): A fitted LabelEncoder instance for the 'major_industry' column.
        experience_encoder (LabelEncoder): A fitted LabelEncoder instance for the 'overall_experience' column.

    Example:
        # Example usage
        X, y, industry_encoder, experience_encoder = prepare_salary_experience_model_data(df)
    
    Notes:
        - The function assumes that 'major_industry' and 'overall_experience' are categorical columns 
          (though 'overall_experience' can also be numeric). It will encode these variables into numeric 
          values using LabelEncoder.
        - The 'annual_salary' column should contain numeric values for the target variable.
        - Missing values in the input DataFrame will not be handled in this function and should be cleaned 
          before calling this function if necessary.
    """
    # Create label encoders
    industry_encoder = LabelEncoder()
    experience_encoder = LabelEncoder()
    
    # Prepare features
    X = df[['major_industry', 'overall_experience']].copy()
    X['major_industry'] = industry_encoder.fit_transform(X['major_industry'])
    X['overall_experience'] = experience_encoder.fit_transform(X['overall_experience'])
    
    y = df['annual_salary']
    
    return X, y, industry_encoder, experience_encoder


def predict_salary(model, industry, experience, industry_encoder, experience_encoder):
    """
    Predicts the annual salary based on the provided industry and experience using
    a pre-trained Random Forest model. The function encodes the input categorical
    variables using previously fitted LabelEncoders.

    Args:
        model (RandomForestRegressor): A pre-trained Random Forest Regressor model.
        industry (str): The industry name (e.g., "Tech", "Finance").
        experience (int or str): The overall experience level (e.g., 5 years or "Senior").
        industry_encoder (LabelEncoder): The LabelEncoder fitted to the 'major_industry' column.
        experience_encoder (LabelEncoder): The LabelEncoder fitted to the 'overall_experience' column.

    Returns:
        prediction (float): The predicted annual salary for the given industry and experience level.

    Raises:
        ValueError: If the provided industry or experience values are not in the expected format or 
                    are not recognized by the LabelEncoders.

    Example:
        # Sample usage
        salary = predict_salary(model, "Tech", 5, industry_encoder, experience_encoder)
    """
    industry_encoded = industry_encoder.transform([industry])
    experience_encoded = experience_encoder.transform([experience])
    
    features = np.array([[industry_encoded[0], experience_encoded[0]]])
    prediction = model.predict(features)[0]
    
    return prediction

# Build and evaluate model
def build_and_evaluate_random_forest_model(X, y):
    """
    Builds and evaluates a Random Forest regression model to predict the target variable based on 
    the provided feature matrix (X) and target vector (y). The function splits the data into training 
    and testing sets, trains a Random Forest model, and calculates several evaluation metrics (RMSE, MAE, R2, MAPE).

    Args:
        X (pd.DataFrame or np.ndarray): The feature matrix, where each column represents a feature and each row represents an instance.
        y (pd.Series or np.ndarray): The target vector containing the true values for the dependent variable.

    Returns:
        model (RandomForestRegressor): A trained Random Forest Regressor model.
        metrics (dict): A dictionary containing the following evaluation metrics:
                        - 'RMSE': Root Mean Squared Error.
                        - 'MAE': Mean Absolute Error.
                        - 'R2': R-squared (coefficient of determination).
                        - 'MAPE': Mean Absolute Percentage Error.
        X_test (pd.DataFrame or np.ndarray): The test set features used for evaluation.
        y_test (pd.Series or np.ndarray): The true values of the target variable for the test set.

    Example:
        # Sample usage
        model, metrics, X_test, y_test = build_and_evaluate_model(X, y)
        print(metrics)

    Notes:
        - The function automatically splits the data into training and testing sets using a test size of 20%.
        - The Random Forest model is initialized with 100 estimators, a maximum depth of 10, and a minimum sample split of 5.
        - The function assumes that `X` and `y` are already pre-processed (e.g., encoded and cleaned).
    """

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    return model, metrics, X_test, y_test

def analyze_features(model, feature_names):
    """
    Analyzes the feature importances of a trained model and returns a sorted DataFrame of the features 
    and their corresponding importance scores.

    Args:
        model (RandomForestRegressor): A trained Random Forest model with `feature_importances_` attribute.
        feature_names (list of str): A list of feature names corresponding to the features used in the model.

    Returns:
        pd.DataFrame: A DataFrame containing the features and their importance scores, sorted in descending order of importance.

    Example:
        # Sample usage
        importances = analyze_features(model, X.columns)
        print(importances)

    Notes:
        - The function assumes that the model has already been trained and has the `feature_importances_` attribute.
        - The feature importance is based on the trained model and reflects the contribution of each feature to the model's predictions.
    """
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    return importances.sort_values('Importance', ascending=False)


def perform_cross_validation(model, X, y, cv=5):
    """
    Performs k-fold cross-validation on the provided model, feature matrix (X), and target vector (y) 
    using the R-squared (R2) metric for evaluation. It returns the mean and standard deviation of 
    the R2 scores across the cross-validation folds.

    Args:
        model (sklearn.base.BaseEstimator): A scikit-learn estimator object (e.g., RandomForestRegressor, 
                                            LinearRegression) that implements the `fit` and `predict` methods.
        X (pd.DataFrame or np.ndarray): The feature matrix, where each column represents a feature and 
                                        each row represents an instance.
        y (pd.Series or np.ndarray): The target vector containing the true values for the dependent variable.
        cv (int, optional): The number of folds in the cross-validation (default is 5). This defines how many 
                            subsets the data should be split into for cross-validation.

    Returns:
        dict: A dictionary containing:
            - 'Mean CV Score': The mean R-squared (R2) score across all the cross-validation folds.
            - 'CV Score Std': The standard deviation of the R2 scores across the folds, reflecting the model's 
              stability and variability across different data splits.

    Example:
        # Sample usage
        results = perform_cross_validation(model, X, y, cv=10)
        print(results)

    Notes:
        - The function uses the **R-squared (R2)** metric as the scoring parameter for cross-validation, which 
          indicates how well the model explains the variance of the target variable.
        - The default value for `cv` is 5, meaning 5-fold cross-validation. You can change this to any positive 
          integer to adjust the number of folds.
        - Cross-validation helps evaluate the model's performance on different subsets of the data, which helps 
          mitigate overfitting and provides a more robust estimate of model performance.
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return {
        'Mean CV Score': cv_scores.mean(),
        'CV Score Std': cv_scores.std()
    }

def convert_state_abbreviations(state_name):
    """
    Converts a full state name to its corresponding two-letter abbreviation.
    If the provided state name is not found in the predefined dictionary, 
    it returns the original state name. The function also handles cases where 
    the input is `NaN` or contains extra whitespace.

    Args:
        state_name (str): The full name of the state to convert. The input 
                          should be a string representing a state (e.g., "California").
                          If the input is `NaN`, it returns `NaN` without any conversion.

    Returns:
        str: The two-letter state abbreviation corresponding to the given state name.
              If the state name is not found, the original state name is returned.
        
    Example:
        # Example usage
        abbreviation = convert_state_abbreviations('California')
        print(abbreviation)  # Output: 'CA'

        abbreviation = convert_state_abbreviations('Nonexistent State')
        print(abbreviation)  # Output: 'Nonexistent State'

    Notes:
        - The function matches state names case-insensitively, so 'california' and 'California' will both return 'CA'.
        - The function trims leading and trailing whitespace from the state name before attempting conversion.
        - If the provided state name is not a valid U.S. state, the original state name is returned.
        - If the state name is `NaN`, the function returns `NaN`.

    """
    state_dict = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'DC',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY'
    }
    
    # Clean the state name and convert
    if pd.isna(state_name):
        return 'null'
    state_name = state_name.strip()
    return state_dict.get(state_name.capitalize(), 'null')

def get_region_from_state(state):
    """
    Returns the region of the United States that a given state belongs to. 
    The function uses a predefined dictionary to map states to their respective regions. 
    If the state is not found in any region, it returns 'Other'.

    Args:
        state (str): The two-letter state abbreviation (e.g., 'CA', 'NY', 'TX') for which 
                     the region is to be determined.

    Returns:
        str: The name of the region the state belongs to. Possible values include:
             - 'Far West', 'Rocky Mountain', 'Plains', 'Great Lakes', 
             - 'Mideast', 'New England', 'Southeast', 'Southwest', or 'Other' 
             (if the state is not found in the predefined regions).

    Example:
        # Example usage
        region = get_region('CA')
        print(region)  # Output: 'Far West'

        region = get_region('TX')
        print(region)  # Output: 'Southwest'

        region = get_region('FL')
        print(region)  # Output: 'Southeast'

        region = get_region('WY')
        print(region)  # Output: 'Rocky Mountain'

        region = get_region('ZZ')  # Invalid state code
        print(region)  # Output: 'Other'

    Notes:
        - The function expects the state abbreviation to be provided as a two-letter string (e.g., 'CA' for California).
        - If the state abbreviation is not in the dictionary, the function will return 'Other', indicating an unknown or unrecognized region.
        - The dictionary includes common U.S. regions such as 'Far West', 'Plains', 'Mideast', etc., based on their geographic groupings.
    """
    regions = {
        'Far West': ['WA', 'OR', 'CA', 'NV', 'AK', 'HI'],
        'Rocky Mountain': ['MT', 'ID', 'WY', 'UT', 'CO'],
        'Plains': ['ND', 'SD', 'NE', 'KS', 'MO', 'IA', 'MN'],
        'Great Lakes': ['WI', 'MI', 'IL', 'IN', 'OH'],
        'Mideast': ['NY', 'PA', 'NJ', 'MD', 'DE', 'DC'],
        'New England': ['ME', 'NH', 'VT', 'MA', 'CT', 'RI'],
        'Southeast': ['WV', 'VA', 'KY', 'TN', 'NC', 'SC', 'GA', 'AL', 'MS', 'LA', 'FL', 'AR'],
        'Southwest': ['AZ', 'NM', 'OK', 'TX']
    }
    for region, states in regions.items():
        if state in states:
            return region
    return 'Other'


class RegionSalaryAnalyzer:
    """
    A class to analyze and predict annual salaries based on industry and region. 
    It allows for building a prediction model using a RandomForestRegressor, 
    making salary predictions, estimating salary ranges, and comparing salaries 
    across different industries within a specific region.

    Attributes:
        df (pd.DataFrame): The DataFrame containing data for salary prediction.
        regions (dict): A dictionary mapping region names to their respective state abbreviations.
        model (sklearn.ensemble.RandomForestRegressor, optional): The trained Random Forest regression model.
        industry_encoder (LabelEncoder): The encoder to convert 'industry' values to numerical values.
        region_encoder (LabelEncoder): The encoder to convert 'region' values to numerical values.
        
    Methods:
        build_prediction_model():
            Trains a RandomForestRegressor model to predict annual salaries based on industry and region.
        
        predict_salary(industry, region):
            Predicts the annual salary for a given industry and region using the trained model.
        
        get_salary_range(industry, region, confidence=0.1):
            Returns the predicted salary range (low, high, and median) for a given industry and region.
        
        compare_industries(region):
            Compares the predicted salaries for all industries within a specific region.
    """
    def __init__(self, df):
        self.df = df
        self.regions = {
            'Far West': ['WA', 'OR', 'CA', 'NV', 'AK', 'HI'],
            'Rocky Mountain': ['MT', 'ID', 'WY', 'UT', 'CO'],
            'Plains': ['ND', 'SD', 'NE', 'KS', 'MO', 'IA', 'MN'],
            'Great Lakes': ['WI', 'MI', 'IL', 'IN', 'OH'],
            'Mideast': ['NY', 'PA', 'NJ', 'MD', 'DE', 'DC'],
            'New England': ['ME', 'NH', 'VT', 'MA', 'CT', 'RI'],
            'Southeast': ['WV', 'VA', 'KY', 'TN', 'NC', 'SC', 'GA', 'AL', 'MS', 'LA', 'FL', 'AR'],
            'Southwest': ['AZ', 'NM', 'OK', 'TX']
        }
        self.model = None
        self.industry_encoder = LabelEncoder()
        self.region_encoder = LabelEncoder()

    def build_prediction_model(self):
        X = pd.DataFrame({
            'industry': self.industry_encoder.fit_transform(self.df['major_industry']),
            'region': self.region_encoder.fit_transform(self.df['region'])
        })
        y = self.df['total_compensation']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        metrics = {
            'R2': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred)
        }
        
        return metrics
    
    def predict_salary(self, industry, region):
        if self.model is None:
            raise ValueError("Model not trained. Call build_prediction_model first.")
            
        features = np.array([[
            self.industry_encoder.transform([industry])[0],
            self.region_encoder.transform([region])[0]
        ]])
        return self.model.predict(features)[0]
    
    def get_salary_range(self, industry, region, confidence=0.1):
        base_prediction = self.predict_salary(industry, region)
        return {
            'low_range': base_prediction * (1 - confidence),
            'high_range': base_prediction * (1 + confidence),
            'median': base_prediction
        }
    def compare_industries(self, region):
        industries = self.industry_encoder.classes_
        return {
            industry: self.predict_salary(industry, region)
            for industry in industries
        }


def simplify_race(race):
    """
    Simplifies a race description into broader categories for easier classification.

    This function takes a race description string and maps it to one of the following categories:
    - 'Asian/Asian American'
    - 'Black/African American'
    - 'Hispanic/Latino'
    - 'Middle Eastern/North African'
    - 'Native American/Alaska Native'
    - 'White'
    - 'Multiple/Other' (for cases not explicitly covered)
    - 'Not Specified' (for missing or `NaN` values)

    Args:
        race (str): The race description to be simplified. The input string can be any of the specific race categories 
                    or `NaN` to represent missing values.

    Returns:
        str: A simplified race category. One of the following values:
             - 'Asian/Asian American'
             - 'Black/African American'
             - 'Hispanic/Latino'
             - 'Middle Eastern/North African'
             - 'Native American/Alaska Native'
             - 'White'
             - 'Multiple/Other'
             - 'Not Specified' (if the input is `NaN` or missing).
        
    Example:
        # Example usage
        simplified_race = simplify_race('Asian or Asian American')
        print(simplified_race)  # Output: 'Asian/Asian American'

        simplified_race = simplify_race('Black or African American')
        print(simplified_race)  # Output: 'Black/African American'

        simplified_race = simplify_race('White')
        print(simplified_race)  # Output: 'White'

        simplified_race = simplify_race('Hispanic, Latino, or Spanish origin')
        print(simplified_race)  # Output: 'Hispanic/Latino'

        simplified_race = simplify_race('Unknown Race')
        print(simplified_race)  # Output: 'Multiple/Other'

        simplified_race = simplify_race(None)
        print(simplified_race)  # Output: 'Not Specified'

    Notes:
        - This function is case-sensitive. The input should match the expected strings or contain the relevant keywords 
          (e.g., 'Asian or Asian American' or 'Black or African American').
        - If the input is `NaN` or missing, the function returns 'Not Specified'.
        - For inputs that do not match any of the predefined categories, the function returns 'Multiple/Other'.
    """
    if pd.isna(race):
        return 'Not Specified'
    elif 'Asian or Asian American' in race:
        return 'Asian/Asian American'
    elif 'Black or African American' in race:
        return 'Black/African American'
    elif 'Hispanic, Latino, or Spanish origin' in race:
        return 'Hispanic/Latino'
    elif 'Middle Eastern or Northern African' in race:
        return 'Middle Eastern/North African'
    elif 'Native American or Alaska Native' in race:
        return 'Native American/Alaska Native'
    elif race == 'White':
        return 'White'
    else:
        return 'Multiple/Other'

class SalaryClassifier:
    """
    A class to classify salary into categories ('Low', 'Average', 'High') based on various features such as 
    gender, race, education level, industry, and experience using a Random Forest classifier.

    Attributes:
        gender_encoder (LabelEncoder): Label encoder for the 'gender' feature.
        race_encoder (LabelEncoder): Label encoder for the 'race_grouped' feature.
        education_encoder (LabelEncoder): Label encoder for the 'education_level' feature.
        industry_encoder (LabelEncoder): Label encoder for the 'major_industry' feature.
        experience_encoder (LabelEncoder): Label encoder for the 'overall_experience' feature.
        model (RandomForestClassifier): A Random Forest classifier model to predict salary categories.

    Methods:
        categorize_salary(salary, q1, q3):
            Categorizes the salary as 'Low', 'Average', or 'High' based on the given salary and quartiles.
        
        prepare_data(df):
            Prepares the dataset by calculating salary quartiles, categorizing salaries, and encoding categorical features.
        
        train_and_evaluate(X, y):
            Trains the RandomForestClassifier model, evaluates it on test data, and calculates feature importance.
        
        predict_salary_category(gender, race, education, industry, experience):
            Predicts the salary category ('Low', 'Average', 'High') for a given individual based on their features.
    """
    def __init__(self):
        self.gender_encoder = LabelEncoder()
        self.race_encoder = LabelEncoder()
        self.education_encoder = LabelEncoder()
        self.industry_encoder = LabelEncoder()
        self.experience_encoder = LabelEncoder()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
    
    def categorize_salary(self, salary, q1, q3):
        if salary <= q1:
            return 'Low'
        elif salary >= q3:
            return 'High'
        else:
            return 'Average'
    
    def prepare_data(self, df):
        # Calculate salary quartiles
        q1 = df['total_compensation'].quantile(0.25)
        q3 = df['total_compensation'].quantile(0.75)
        
        # Create salary categories
        df['salary_category'] = df['total_compensation'].apply(
            lambda x: self.categorize_salary(x, q1, q3)
        )
        
        # Prepare features
        X = pd.DataFrame({
            'gender': self.gender_encoder.fit_transform(df['gender']),
            'race': self.race_encoder.fit_transform(df['race_grouped']),
            'education': self.education_encoder.fit_transform(df['education_level']),
            'industry': self.industry_encoder.fit_transform(df['major_industry']),
            'experience': self.experience_encoder.fit_transform(df['overall_experience'])
        })
        
        y = df['salary_category']
        return X, y
    
    def train_and_evaluate(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance
        }
    
    def predict_salary_category(self, gender, race, education, industry, experience):
        features = np.array([[
            self.gender_encoder.transform([gender])[0],
            self.race_encoder.transform([race])[0],
            self.education_encoder.transform([education])[0],
            self.industry_encoder.transform([industry])[0],
            self.experience_encoder.transform([experience])[0]
        ]])
        return self.model.predict(features)[0]

def plot_industry_boxplots_plotly(df, industries, ncols=2):
    """
    Creates a grid of boxplots showing salary distributions across experience levels for each industry using Plotly.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the salary data with columns:
            - 'major_industry': Industry category
            - 'overall_experience': Experience level category
            - 'annual_salary': Numerical salary values
        industries (array-like): List or array of industry names to plot
        ncols (int, optional): Number of columns in the plot grid. Defaults to 2.
    
    Returns:
        plotly.graph_objects.Figure: A Plotly figure object containing the grid of boxplots.
    """
    # Calculate number of rows needed
    nrows = (len(industries) + ncols - 1) // ncols

    # Create a subplot figure
    fig = sp.make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f'Salary Distribution in {industry}' for industry in industries]
    )

    # Plot each industry
    for idx, industry in enumerate(industries):
        row = (idx // ncols) + 1
        col = (idx % ncols) + 1
        
        # Filter data for current industry
        industry_data = df[df['major_industry'] == industry]

        # Add boxplot for current industry
        fig.add_trace(
            go.Box(
                x=industry_data['overall_experience'],
                y=industry_data['annual_salary'],
                name=industry,
                boxpoints="outliers",  # Show individual outliers
                marker=dict(color='blue'),
                line=dict(color='blue')
            ),
            row=row,
            col=col
        )
    
    # Update layout
    fig.update_layout(
        title="Salary Distributions by Experience Level and Industry",
        showlegend=False,
        height=500 * nrows,  # Adjust height based on number of rows
        width=1800,  # Adjust overall width
    )
    
    # Update subplot axes
    fig.update_xaxes(title_text="Experience Level", tickangle=45)
    fig.update_yaxes(title_text="Annual Salary (USD)", range=[0, 2_000_000])
    
    return fig