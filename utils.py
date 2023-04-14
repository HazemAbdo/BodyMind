categorical_attribuites = [
    "Gender",
    "H_Cal_Consump",
    "Alcohol_Consump",
    "Smoking",
    "Food_Between_Meals",
    "Fam_Hist",
    "H_Cal_Burn",
    "Transport",
    "Body_Level"
]
attribuite_description = {
    "Gender": "Male or female.",
    "Age": "Numeric value.",
    "Height": "Numeric value (in meters).",
    "Weight": "Numeric value (in kilograms).",
    "Fam_Hist": "Does the family have a history with obesity?",
    "H_Cal_Consump": "High caloric food consumption.",
    "Veg_Consump": "Frequency of vegetables consumption.",
    "Meal_Count": "Average number of meals per day.",
    "Food_Between_Meals": "Frequency of eating between meals.",
    "Smoking": "Is the person smoking?",
    "Water_Consump": "Frequency of water consumption.",
    "H_Cal_Burn": "Does the body have high calories burn rate?",
    "Phys_Act": "How often does the person do physical activities?",
    "Time_E_Dev": "How much time does person spend on electronic devices.",
    "Alcohol_Consump": "Frequency of alcohols consumption.",
    "Transport": "Which transports does the person usually use?",
}
code_value = {
    "Body_Level": {
        "Body Level 1": 1,
        "Body Level 2": 2,
        "Body Level 3": 3,
        "Body Level 4": 4,
    },
    "Gender": {
        "Female": 0,
        "Male": 1
    },
    "H_Cal_Consump": {
        "no": 0,
        "yes": 1
    },
    "Alcohol_Consump": {
        "no": 0,
        "Sometimes": 1,
        "Frequently": 2,
        "Always": 3
    },
    "Smoking": {
        "no": 0,
        "yes": 1
    },
    "Food_Between_Meals": {
        "no": 0,
        "Sometimes": 1,
        "Frequently": 2,
        "Always": 3
    },
    "Fam_Hist": {
        "no": 0,
        "yes": 1
    },
    "H_Cal_Burn": {
        "no": 0,
        "yes": 1
    },
    "Transport": {
        "Public_Transportation": 0,
        "Automobile": 1,
        "Walking": 2,
        "Bike": 3,
        "Motorbike": 4
    }
}
def describe_attribuite(attr):
    print(f"{attr}: {attribuite_description[attr]}")
def is_categorical(attr):
    return attr in categorical_attribuites
def encode_categorical_features(df):
    df_encoded = df.copy()
    for col in df_encoded.columns:
        #print original column name and new column name
        if is_categorical(col):
            if col in code_value.keys():
                df_encoded[col] = df_encoded[col].map(code_value[col])
    return df_encoded
def read_data(file_name='body_level_classification_train.csv'):
    '''
    read data from csv file into pandas dataframe
    labeling categorical data as category type
    separate attributes and classes
    and encode categorical features
    '''
    df = pd.read_csv(file_name)
    df=encode_categorical_features(df)
    attributes = df[df.columns[:-1]]
    class_1_df=df[df['Body_Level']==1].drop(columns=['Body_Level'])
    class_2_df=df[df['Body_Level']==2].drop(columns=['Body_Level'])
    class_3_df=df[df['Body_Level']==3].drop(columns=['Body_Level'])
    class_4_df=df[df['Body_Level']==4].drop(columns=['Body_Level'])
    return df,attributes,class_1_df,class_2_df,class_3_df,class_4_df