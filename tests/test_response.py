from fastapi.testclient import TestClient
from app import app

def train_success():
    endpoint = '/v1/census/train'
    body = { 
        "link":"https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"
    }

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 200

def test_bad_request():
    endpoint = '/v1/census/train'
    body = { "link": "asjkbas" }

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 400

def predict_success():
    endpoint = '/v1/census/predict'
    body = {
            "dataArray": [{
                "age": 50,
                "worker_class": " Self-employed-not incorporated",
                "industry_code": 4,
                "occupation_code": 34,
                "education": " High school graduate",
                "wage_per_hour": 0,
                "enroll_edu_last_wk": " Not in universe",
                "marital_status": " Married-civilian spouse present",
                "major_industry_code": " Construction",
                "major_occupation_code": " Precision production craft & repair",
                "race": " White",
                "hispanic_origin": " All other",
                "sex": " Male",
                "labor_union_member": " Not in universe",
                "unemployment_reason": " Not in universe",
                "employment_status": " Full-time schedules",
                "capital_gains": 7298,
                "capital_losses": 0,
                "stock_dividends": 0,
                "tax_filer_status": " Joint both under 65",
                "previous_residence_region": " Not in universe",
                "previous_residence_state": " Not in universe",
                "household_status": " Householder",
                "household_summary": " Householder",
                "instance_weight": 1038.34,
                "migration_code_change_in_msa": " ?",
                "migration_code_change_in_reg": " ?",
                "migration_code_move_within_reg": " ?",
                "live_in_this_house_1_year_ago": " Not in universe under 1 year old",
                "migration_prev_res_in_sunbelt": " ?",
                "num_persons_worked_for_employer": 1,
                "family_members_under_18": " Not in universe",
                "birth_country_father": " ?",
                "birth_country_mother": " ?",
                "birth_country": " ?",
                "citizenship": " Foreign born- U S citizen by naturalization",
                "self_employed": 1,
                "veteran_questionnaire": " Not in universe",
                "veteran_benfits": 2,
                "weeks_worked_in_year": 52,
                "year": 95
            }]
        }

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 200


def test_predict_bad_request():
    endpoint = '/v1/census/predict'
    body = { "link": "asjkbas" }

    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 422