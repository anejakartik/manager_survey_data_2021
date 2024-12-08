import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


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
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return {
        'Mean CV Score': cv_scores.mean(),
        'CV Score Std': cv_scores.std()
    }