PROTECTED = {"CODE_GENDER_M", "DAYS_BIRTH", "NAME_FAMILY_STATUS_Married"}

NOT_REPORTED = {
    "OWN_CAR_AGE",                    # 66% missing, weak, not actionable
    "DEF_30_CNT_SOCIAL_CIRCLE",       # other people's behaviour
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "APARTMENTS_MEDI",                # building statistics
    "YEARS_BEGINEXPLUATATION_MODE",
    "OBS_30_CNT_SOCIAL_CIRCLE",       # other people's behaviour
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "APARTMENTS_MODE",                # building statistics
    "LIVINGAPARTMENTS_AVG",
    "REGION_POPULATION_RELATIVE",
    "NAME_EDUCATION_TYPE_Higher_education",
    "NAME_EDUCATION_TYPE_Secondary___secondary_special",
    "NAME_EDUCATION_TYPE_Lower_secondary",
    "NAME_EDUCATION_TYPE_Incomplete_higher",
    "FLAG_DOCUMENT_3",
    "FLAG_WORK_PHONE",
    "HOUR_APPR_PROCESS_START",
}

EXCLUDED = PROTECTED | NOT_REPORTED

REASONS = {
    "EXT_SOURCE_1": "Low external credit score (source 1)",
    "EXT_SOURCE_2": "Low external credit score (source 2)",
    "EXT_SOURCE_3": "Low external credit score (source 3)",
    "AMT_CREDIT": "Loan amount high relative to goods price",
    "AMT_GOODS_PRICE": "Loan amount high relative to goods price",
    "AMT_ANNUITY": "High loan repayment burden",
    "AMT_INCOME_TOTAL": "Income level relative to loan",
    "DAYS_EMPLOYED": "Short or no employment history",
    "NAME_INCOME_TYPE_Working": "Income source type",
    "ORGANIZATION_TYPE_Self_employed": "Self-employment income stability",
    "ORGANIZATION_TYPE_Business_Entity_Type_3": "Employer industry",
    "OCCUPATION_TYPE_Laborers": "Occupation type",
    "OCCUPATION_TYPE_Sales_staff": "Occupation type",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Many recent credit enquiries",
    "NAME_CONTRACT_TYPE_Revolving_loans": "Loan product type",
    "DAYS_ID_PUBLISH": "Recent changes to identity/registration details",
    "DAYS_REGISTRATION": "Recent changes to identity/registration details",
    "REGION_RATING_CLIENT_W_CITY": "Residential region risk rating",
    "REG_CITY_NOT_LIVE_CITY": "Registered address differs from city of residence",
    "DAYS_LAST_PHONE_CHANGE": "Recent phone number change",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Many recent credit enquiries",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Many recent credit enquiries",
    "EXT_SOURCE_1": "Low external credit score",
    "EXT_SOURCE_2": "Low external credit score",
    "EXT_SOURCE_3": "Low external credit score",
    "ORGANIZATION_TYPE_Construction": "Employer industry",
    "ORGANIZATION_TYPE_Transport__type_3": "Employer industry",
    "OCCUPATION_TYPE_Drivers": "Occupation type",
    "OCCUPATION_TYPE_Low_skill_Laborers": "Occupation type",
}

BUILDING_SUFFIXES = ("_AVG", "_MODE", "_MEDI")

def is_excluded(feat):
    return feat in EXCLUDED or feat.endswith(BUILDING_SUFFIXES)