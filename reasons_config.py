PROTECTED = {"CODE_GENDER_M", "DAYS_BIRTH", "NAME_FAMILY_STATUS_Married"}

NOT_REPORTED = {
    "OWN_CAR_AGE",                    # 66% missing, weak, not actionable
    "DEF_30_CNT_SOCIAL_CIRCLE",       # other people's behaviour
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "APARTMENTS_MEDI",                # building statistics
    "YEARS_BEGINEXPLUATATION_MODE",
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
    "NAME_EDUCATION_TYPE_Higher_education": "Education level",
    "NAME_EDUCATION_TYPE_Secondary___secondary_special": "Education level",
    "NAME_CONTRACT_TYPE_Revolving_loans": "Loan product type",
    "DAYS_ID_PUBLISH": "Recent changes to identity/registration details",
    "DAYS_REGISTRATION": "Recent changes to identity/registration details",
    "FLAG_DOCUMENT_3": "Documentation provided",
    "FLAG_WORK_PHONE": "Contact details provided",
    "REGION_RATING_CLIENT_W_CITY": "Residential region risk rating",
    "REG_CITY_NOT_LIVE_CITY": "Registered address differs from city of residence",
    "DAYS_LAST_PHONE_CHANGE": "Recent phone number change",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Many recent credit enquiries",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Many recent credit enquiries",
}