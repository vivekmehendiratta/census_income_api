from ast import Str
from pydantic import BaseModel, conlist
from typing import List, Any
# class CensusRaw(BaseModel):
#     data: str
#     link: str

class CensusDataPoint(BaseModel):
    age: int
    worker_class: str
    industry_code: int
    occupation_code: int
    education: str
    wage_per_hour:int
    enroll_edu_last_wk:str
    marital_status:str
    major_industry_code:str
    major_occupation_code:str
    race:str
    hispanic_origin:str
    sex:str
    labor_union_member:str
    unemployment_reason:str
    employment_status:str
    capital_gains:float
    capital_losses:float
    stock_dividends:float
    tax_filer_status:str
    previous_residence_region:str
    previous_residence_state:str
    household_status:str
    instance_weight:float
    household_summary:str
    migration_code_change_in_msa:str
    migration_code_change_in_reg:str
    migration_code_move_within_reg:str
    live_in_this_house_1_year_ago:str
    migration_prev_res_in_sunbelt:str
    num_persons_worked_for_employer:int
    family_members_under_18:str
    birth_country_father:str
    birth_country_mother:str
    birth_country:str
    citizenship:str
    self_employed:int
    veteran_questionnaire:str
    veteran_benfits:int
    weeks_worked_in_year:int
    year:int

class CensusDataPointList(BaseModel):
    dataArray: List[CensusDataPoint]

class CensusLink(BaseModel):
    # train path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz'
    # test path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz'
    link: str

class CensusTrainingResponse(BaseModel):
    Precision:float
    Recall: float
    F1Score: float

class CensusPredictionResponse(BaseModel):
    Precision:float
    Recall: float
    F1Score: float
    Accuracy: float

class CensusJsonPredictionResponse(BaseModel):
    prediction: List[int]