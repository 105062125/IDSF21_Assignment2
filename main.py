from matplotlib import figure
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression



def initialize_sidebar():
    st.title('Employ Attrition')
    st.sidebar.markdown('# About this project:')
    st.sidebar.markdown('This interactive website can help you \
        uncover the factors that lead to employee attrition. Enjoy!')
    st.sidebar.markdown('## What you can do:')
    st.sidebar.markdown('1. Find out the correlation in different categories\
        lead to employee attrition')
    st.sidebar.markdown('2. Explore the relation between each features')
    st.sidebar.markdown('3. Predict if you want to quit the job')
    st.sidebar.markdown('## Data:')
    st.sidebar.markdown('The dataset contains 1,470 employees with 35 \
        different attributes like age, education, \
            distance from home and work-life balance.')
    st.sidebar.markdown('Link: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset')
    st.sidebar.markdown('## Author:')
    st.sidebar.markdown('Kevin Chen')


    # st.sidebar.caption('Kevin Chen @ 2021')
@st.cache
def load_data():
    
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    col = df.columns.tolist()
    col[0], col[1] = col[1], col[0]
    df = df.reindex(columns=col)
    df.Attrition = pd.Categorical(df.Attrition)
    df['Attrition code'] = df.Attrition.cat.codes
    # df['Attrition code'] = df.Attrition.apply(lambda v: 1 if v=='Yes' else 0)
    df.BusinessTravel = pd.Categorical(df.BusinessTravel)
    df['BusinessTravel code'] = df.BusinessTravel.cat.codes
    df.Department = pd.Categorical(df.Department)
    df['Department code'] = df.BusinessTravel.cat.codes
    df.EducationField = pd.Categorical(df.EducationField)
    df['EducationField code'] = df.EducationField.cat.codes
    df.Gender = pd.Categorical(df.Gender)
    df['Gender code'] = df.Gender.cat.codes
    df.JobRole = pd.Categorical(df.JobRole)
    df['JobRole code'] = df.JobRole.cat.codes
    df.MaritalStatus = pd.Categorical(df.MaritalStatus)
    df['MaritalStatus code'] = df.MaritalStatus.cat.codes
    df.OverTime = pd.Categorical(df.OverTime)
    df['OverTime code'] = df.OverTime.cat.codes
    
    return df

@st.cache
def corr_data(df):
    df.Attrition = pd.Categorical(df.Attrition)
    df['Attrition code'] = df.Attrition.cat.codes
    # df['Attrition code'] = df.Attrition.apply(lambda v: 1 if v=='Yes' else 0)
    df.BusinessTravel = pd.Categorical(df.BusinessTravel)
    df['BusinessTravel code'] = df.BusinessTravel.cat.codes
    df.Department = pd.Categorical(df.Department)
    df['Department code'] = df.BusinessTravel.cat.codes
    df.EducationField = pd.Categorical(df.EducationField)
    df['EducationField code'] = df.EducationField.cat.codes
    df.Gender = pd.Categorical(df.Gender)
    df['Gender code'] = df.Gender.cat.codes
    df.JobRole = pd.Categorical(df.JobRole)
    df['JobRole code'] = df.JobRole.cat.codes
    df.MaritalStatus = pd.Categorical(df.MaritalStatus)
    df['MaritalStatus code'] = df.MaritalStatus.cat.codes
    df.OverTime = pd.Categorical(df.OverTime)
    df['OverTime code'] = df.OverTime.cat.codes
    
    df_drop = df.drop(columns=['EmployeeCount','StandardHours', 'EmployeeNumber', 'Over18'])
    first_col = df_drop.pop('Attrition code')
    df_drop.insert(0, 'Attrition code', first_col)
    corr = df_drop.corr()

    
    

    return corr
    
@st.cache
def plot_specific_category(df,abs_check,choose_index):
    
    
    new_df = pd.DataFrame(df[choose_index])
    new_df = new_df.drop([choose_index])
    if abs_check: new_df = new_df.apply(lambda v: abs(v))
    new_df = new_df.sort_values(by=choose_index, ascending=False)

    return new_df
    

    # sns.barplot(y = new_df, color="goldenrod", ax=ax)
    # ax.set_ylabel('Percentage')
    # ax.set_xlabel('Gender')
    # st.pyplot(fig)
    # st.write(new_df)
    # fig, ax = plt.subplots(figsize=(20, 12))
    # ax = new_df.plot.bar(y='Attrition code')
    # st.bar_chart(new_df)

@st.cache(allow_output_mutation=True)
def train():
    data = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    data.drop(['EmployeeNumber','DailyRate','MonthlyRate','HourlyRate','PercentSalaryHike','PerformanceRating','TrainingTimesLastYear','JobLevel'],axis=1,inplace=True)

    data['Attrition'] = LabelEncoder().fit_transform(data['Attrition'])
    data["BusinessTravel"] = LabelEncoder().fit_transform(data['BusinessTravel'])
    data["Department"] = LabelEncoder().fit_transform(data['Department'])
    data["EducationField"] = LabelEncoder().fit_transform(data['EducationField'])
    data["Gender"] = LabelEncoder().fit_transform(data['Gender'])
    data["JobRole"] = LabelEncoder().fit_transform(data['JobRole'])
    data["MaritalStatus"] = LabelEncoder().fit_transform(data['MaritalStatus'])
    data["Over18"] = LabelEncoder().fit_transform(data['Over18'])
    data["OverTime"] = LabelEncoder().fit_transform(data['OverTime'])
    cols = list(data.columns)
    cols.remove("Attrition")
    sampled,target = SMOTE().fit_resample(data[cols],data["Attrition"])
    X_train,X_test,Y_train,Y_test = train_test_split(sampled[cols],
                                                 target,
                                                 test_size = 0.3,
                                                 shuffle=True)
    logistic_model = LogisticRegression(solver='liblinear',random_state=0).fit(X_train,Y_train)
    random_forest = RandomForestClassifier(n_estimators=500,
                                       random_state=0).fit(X_train,Y_train)
    
    return random_forest, logistic_model

def predict(random_model, logistic_model):
    st.subheader('Try to predict if you want to quit your job')
    st.markdown("This random forest model is around 90% accuracy and logistic regression model \
        is around 82% accuracy. Have fun!")
    st.markdown("Your personal data won't be stored, it is only for prediction. Feel free to use it")
    st.markdown("The only reason I used logistic regression model is because it can show the precentage.")
    st.markdown("Have fun!!!!")
    st.info("Note: It's recommended to choose logistic regression model since it can show the percentage, which is much more fun.\
            Although its accuracy rate is lower than random forest")
    with st.form(key='my_form'):
        MODEL = st.selectbox('Model', ['Logistic Regression','Random Forest'])
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        Age = col1.slider('Age', 1, 100, 30)
        BusinessTravel = col2.selectbox('BusinessTravel', ['Non-Travel','Rarely','Frequent'])
        if BusinessTravel == 'Non-Travel': BusinessTravel= 0
        elif BusinessTravel == 'Frequent': BusinessTravel= 1
        else: BusinessTravel = 2

        Department = col3.selectbox('Department', ['HR','R&D','Sales'])
        if Department == 'HR': Department= 0
        elif Department == 'R&D': Department= 1
        else: Department = 2

        DistanceFromHome = col4.number_input('DistanceFromHome (km)')
        Education = col5.selectbox('Education', ['Below College','College','Bachelor', 'Master', 'Doctor'])
        if Education == 'Below College': Education= 1
        elif Education == 'College': Education= 2
        elif Education == 'Bachelor': Education= 3
        elif Education == 'Master': Education= 4
        else: Education = 5
        
        EducationField = col6.selectbox('EducationField', ['HR','Life science','Marketing', 'Medical','technical','others'])
        if EducationField == 'HR': EducationField= 1
        elif EducationField == 'Life science': EducationField= 2
        elif EducationField == 'Marketing': EducationField= 3
        elif EducationField == 'Medical': EducationField= 4
        elif EducationField == 'others': EducationField= 5
        else: EducationField = 6

        EmployeeCount = 1

        EnvironmentSatisfaction = col7.selectbox('EnvironmentSatisfaction', ['Low','Medium','High', 'Very High'])
        if EnvironmentSatisfaction == 'Low': EnvironmentSatisfaction= 1
        elif EnvironmentSatisfaction == 'Medium': EnvironmentSatisfaction= 2
        elif EnvironmentSatisfaction == 'High': EnvironmentSatisfaction= 3
        elif EnvironmentSatisfaction == 'Very High': EnvironmentSatisfaction= 4

        Gender = col1.selectbox('Gender', ['Male','Female'])
        if Gender == 'Male': Gender= 1
        elif Gender == 'Female': Gender= 0


        JobInvolvement = col2.selectbox('JobInvolvement', ['Low','Medium','High', 'Very High'])
        if JobInvolvement == 'Low': JobInvolvement= 1
        elif JobInvolvement == 'Medium': JobInvolvement= 2
        elif JobInvolvement == 'High': JobInvolvement= 3
        elif JobInvolvement == 'Very High': JobInvolvement= 4
        else: JobInvolvement= 2

        JobRole = col3.selectbox('JobRole', ['HC REP','HR','LAB TECHNICIAN', 'MANAGER', 'MANAGING DIRECTOR', 'REASEARCH DIRECTOR','RESEARCH SCIENTIST','SALES EXECUTIEVE','SALES REPRESENTATIVE'])
        if JobRole == 'HC REP': JobRole= 1
        elif JobRole == 'HR': JobRole = 2
        elif JobRole == 'LAB TECHNICIAN': JobRole= 3
        elif JobRole == 'MANAGER': JobRole= 4
        elif JobRole == 'MANAGING DIRECTOR': JobRole= 5
        elif JobRole == 'REASEARCH DIRECTOR': JobRole= 6
        elif JobRole == 'RESEARCH SCIENTIST': JobRole= 7
        elif JobRole == 'SALES EXECUTIEVE': JobRole= 8
        elif JobRole == 'SALES REPRESENTATIVE': JobRole= 9
        else: JobRole= 3

        JobSatisfaction = col4.selectbox('JobSatisfaction', ['Low','Medium','High', 'Very High'])
        if JobSatisfaction == 'Low': JobSatisfaction= 1
        elif JobSatisfaction == 'Medium': JobSatisfaction= 2
        elif JobSatisfaction == 'High': JobSatisfaction= 3
        elif JobSatisfaction == 'Very High': JobSatisfaction= 4
        else: JobSatisfaction= 2

        MaritalStatus = col5.selectbox('MaritalStatus', ['Single','Married'])
        if MaritalStatus == 'Married': MaritalStatus= 1
        elif MaritalStatus == 'Single': MaritalStatus= 2
        
        MonthlyIncome = col6.number_input('MonthlyIncome')
        NumCompaniesWorked = col1.number_input('NumCompaniesWorked')

        Over18 = col7.selectbox('Over18', ['Yes','No'])
        if Over18 == 'Yes': Over18= 0
        elif Over18 == 'No': Over18= 1

        OverTime = col2.selectbox('OverTime', ['Yes','No'])
        if OverTime == 'Yes': OverTime= 1
        elif OverTime == 'No': OverTime= 0

        RelationshipSatisfaction = col3.selectbox('RelationshipSatisfaction', ['Low','Medium','High', 'Very High'])
        if RelationshipSatisfaction == 'Low': RelationshipSatisfaction= 1
        elif RelationshipSatisfaction == 'Medium': RelationshipSatisfaction= 2
        elif RelationshipSatisfaction == 'High': RelationshipSatisfaction= 3
        elif RelationshipSatisfaction == 'Very High': RelationshipSatisfaction= 4
        else: RelationshipSatisfaction= 2

        StandardHours = 80

        StockOptionLevel = col4.selectbox('StockOptionLevel', [0,1,2,3])
        TotalWorkingYears = col5.number_input('TotalWorkingYears')

        WorkLifeBalance = col6.selectbox('WorkLifeBalance', ['Bad','Good','Better', 'Best'])
        if WorkLifeBalance == 'Bad': WorkLifeBalance= 1
        elif WorkLifeBalance == 'Good': WorkLifeBalance= 2
        elif WorkLifeBalance == 'Better': WorkLifeBalance= 3
        elif WorkLifeBalance == 'Best': WorkLifeBalance= 4
        else: WorkLifeBalance= 2

        YearsAtCompany = col7.number_input('YearsAtCompany')
        YearsInCurrentRole = col1.number_input('YearsInCurrentRole')
        YearsSinceLastPromotion = col2.number_input('YearsSinceLastPromotion')
        YearsWithCurrManager = col3.number_input('YearsWithCurrManager')
        # text_input = st.text_input(label='Enter some text')
        # MODEL = col4.selectbox('Model', ['Logistic Regression','Random Forest'])
        submit_button = st.form_submit_button(label='Predict')
    # ['Age', 'BusinessTravel', 'Department', 'DistanceFromHome', 'Education',
    #    'EducationField', 'EmployeeCount', 'EnvironmentSatisfaction', 'Gender',
    #    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    #    'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked', 'Over18',
    #    'OverTime', 'RelationshipSatisfaction', 'StandardHours',
    #    'StockOptionLevel', 'TotalWorkingYears', 'WorkLifeBalance',
    #    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    #    'YearsWithCurrManager']
    



    sample_X = [[Age, BusinessTravel, Department, DistanceFromHome, Education, EducationField, EmployeeCount, EnvironmentSatisfaction\
        , Gender, JobInvolvement, JobRole, JobSatisfaction, MaritalStatus, MonthlyIncome, NumCompaniesWorked\
            , Over18, OverTime, RelationshipSatisfaction, StandardHours, StockOptionLevel, TotalWorkingYears, WorkLifeBalance\
                , YearsAtCompany ,YearsInCurrentRole, YearsSinceLastPromotion ,YearsWithCurrManager]]
    
    if submit_button: 
        
        # st.info("Note: the percentage is only for reference, the result is from random forest model")
        if MODEL=='Logistic Regression':
            prob = logistic_model.predict_proba(sample_X)[0]
            logistic_predict = logistic_model.predict(sample_X)[0]
            if  logistic_predict == 1:
                st.write("Not quit: ", round(prob[0]*100,2), "%,  Quit: ", round(prob[1]*100,2),"%")
                st.write("Oh! You have ",round(prob[1]*100,2), "% to quit your job")
            else:
                st.write("Not quit: ", round(prob[0]*100,2), "%,  Quit: ", round(prob[1]*100,2), "%")
                st.write("Ha! You have ", round(prob[0]*100,2), "% won't quit your job")
        elif MODEL=='Random Forest':
            random_predict = random_model.predict(sample_X)[0]
            if random_predict == 1:
                st.write("Oh! you will quit your job")
            else:
                st.write("Ha! you won't quit your job")

if __name__ == "__main__":
    # ===========
    # ===config==
    # ===========
    st.set_page_config(layout="wide")
    initialize_sidebar()

    # =============
    # ===raw data==
    # =============
    st.subheader('Raw data')
    df = load_data()
    st.write(df)
    st.markdown('Important Note: ')
    st.write("Education: 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'")
    st.write("EnvironmentSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'")
    st.write("JobInvolvement: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'")
    st.write("JobSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'")
    st.write("PerformanceRating: 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'")
    st.write("WorkLifeBalance: 1 'Bad' 2 'Good' 3 'Better' 4 'Best'")
    st.write("RelationshipSatisfaction: 1 'Low' 2 'Medium' 3 'High' 4 'Very High'")

    # ===========
    # ===corr====
    # ===========
    st.subheader('The correlation coefficient between each categories')
    st.markdown('In this plot, we change the categorical feature into number.\
        For instance, Attrition will become Attrition code (You can check it in the raw data)')
    st.markdown('Set the threshold to see the correlation in this range')
    df_code = corr_data(df)
    minimum, maximum = st.slider("Threshold Range", -1.0, 1.0, (-0.5, 0.5))
    labels = df_code.applymap(lambda v: str(round(v,2)) if maximum >= v >= minimum else "")
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(df_code.corr(), ax=ax, annot=labels, cmap='coolwarm',fmt="s")
    st.pyplot(fig)

    # ===================
    # ===plot category===
    # ===================
    st.subheader('Look closely in specific attribute')
    st.markdown('Choose an attribute to see the sorted correlation.')
    abs_check = st.checkbox('Absolute value')
    choose_index = st.selectbox('Choose an attribute', df_code.index)
    new_df = plot_specific_category(df_code,abs_check,choose_index)
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.barplot(x = new_df.index, y = new_df[choose_index])
    plt.xticks(rotation=60)
    st.write(fig)
    st.markdown('Note:')
    st.markdown('1. Please don\'t let your employee work overtime! They will probably quit their job!!\
    The salary is not the main reason!!')
    st.markdown('2. The higher the hourly rate (salary per hour), the higher the job satisfaction')
    st.markdown('3. Try to find out more interesting things!')

    # ===================
    # ===train model=====
    # ===================

    random_model, logistic_model = train()
    predict(random_model, logistic_model)

    