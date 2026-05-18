import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv('employee_attrition_dataset.csv')

# -----------------------------
# PREPROCESS DATA
# -----------------------------
data = pd.get_dummies(data, drop_first=True)

# -----------------------------
# FEATURES AND TARGET
# -----------------------------
X = data.drop('Attrition_Yes', axis=1)
y = data['Attrition_Yes']

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Employee Attrition Prediction")

st.write("Predict whether an employee will leave the company.")

# -----------------------------
# USER INPUTS
# -----------------------------
Employee_ID = st.number_input("Employee ID", min_value=1)

Age = st.slider("Age", 18, 60)

Job_Level = st.selectbox(
    "Job Level",
    [1, 2, 3, 4, 5]
)

Monthly_Income = st.number_input(
    "Monthly Income",
    min_value=1000
)

Hourly_Rate = st.number_input(
    "Hourly Rate",
    min_value=1
)

Years_at_Company = st.slider(
    "Years at Company",
    0,
    40
)

Years_in_Current_Role = st.slider(
    "Years in Current Role",
    0,
    20
)

Years_Since_Last_Promotion = st.slider(
    "Years Since Last Promotion",
    0,
    15
)

Work_Life_Balance = st.selectbox(
    "Work Life Balance",
    [1, 2, 3, 4]
)

Job_Satisfaction = st.selectbox(
    "Job Satisfaction",
    [1, 2, 3, 4]
)

Performance_Rating = st.selectbox(
    "Performance Rating",
    [1, 2, 3, 4]
)

Training_Hours_Last_Year = st.slider(
    "Training Hours Last Year",
    0,
    100
)

Project_Count = st.slider(
    "Project Count",
    1,
    20
)

Average_Hours_Worked_Per_Week = st.slider(
    "Average Hours Worked Per Week",
    20,
    80
)

Absenteeism = st.slider(
    "Absenteeism",
    0,
    30
)

Work_Environment_Satisfaction = st.selectbox(
    "Work Environment Satisfaction",
    [1, 2, 3, 4]
)

Relationship_with_Manager = st.selectbox(
    "Relationship with Manager",
    [1, 2, 3, 4]
)

Job_Involvement = st.selectbox(
    "Job Involvement",
    [1, 2, 3, 4]
)

Distance_From_Home = st.slider(
    "Distance From Home",
    1,
    50
)

Number_of_Companies_Worked = st.slider(
    "Number of Companies Worked",
    0,
    10
)

# -----------------------------
# CATEGORICAL INPUTS
# -----------------------------
Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Marital_Status = st.selectbox(
    "Marital Status",
    ["Married", "Single", "Other"]
)

Department = st.selectbox(
    "Department",
    ["HR", "IT", "Marketing", "Sales", "Other"]
)

Job_Role = st.selectbox(
    "Job Role",
    ["Assistant", "Executive", "Manager", "Other"]
)

Overtime = st.selectbox(
    "Overtime",
    ["Yes", "No"]
)

# -----------------------------
# CONVERT INPUTS TO MODEL FORMAT
# -----------------------------
input_data = pd.DataFrame({
    'Employee_ID': [Employee_ID],
    'Age': [Age],
    'Job_Level': [Job_Level],
    'Monthly_Income': [Monthly_Income],
    'Hourly_Rate': [Hourly_Rate],
    'Years_at_Company': [Years_at_Company],
    'Years_in_Current_Role': [Years_in_Current_Role],
    'Years_Since_Last_Promotion': [Years_Since_Last_Promotion],
    'Work_Life_Balance': [Work_Life_Balance],
    'Job_Satisfaction': [Job_Satisfaction],
    'Performance_Rating': [Performance_Rating],
    'Training_Hours_Last_Year': [Training_Hours_Last_Year],
    'Project_Count': [Project_Count],
    'Average_Hours_Worked_Per_Week': [Average_Hours_Worked_Per_Week],
    'Absenteeism': [Absenteeism],
    'Work_Environment_Satisfaction': [Work_Environment_Satisfaction],
    'Relationship_with_Manager': [Relationship_with_Manager],
    'Job_Involvement': [Job_Involvement],
    'Distance_From_Home': [Distance_From_Home],
    'Number_of_Companies_Worked': [Number_of_Companies_Worked],

    'Gender_Male': [1 if Gender == 'Male' else 0],

    'Marital_Status_Married': [1 if Marital_Status == 'Married' else 0],
    'Marital_Status_Single': [1 if Marital_Status == 'Single' else 0],

    'Department_HR': [1 if Department == 'HR' else 0],
    'Department_IT': [1 if Department == 'IT' else 0],
    'Department_Marketing': [1 if Department == 'Marketing' else 0],
    'Department_Sales': [1 if Department == 'Sales' else 0],

    'Job_Role_Assistant': [1 if Job_Role == 'Assistant' else 0],
    'Job_Role_Executive': [1 if Job_Role == 'Executive' else 0],
    'Job_Role_Manager': [1 if Job_Role == 'Manager' else 0],

    'Overtime_Yes': [1 if Overtime == 'Yes' else 0]
})

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Attrition"):

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"Employee is likely to leave the company")

    else:
        st.success(f"Employee is likely to stay")

    st.write(f"Attrition Probability: {probability:.2f}")