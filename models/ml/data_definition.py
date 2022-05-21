train_cols = ['age', 'worker_class', 'industry_code', 'occupation_code', 'education',
       'wage_per_hour', 'enroll_edu_last_wk', 'marital_status',
       'major_industry_code', 'major_occupation_code', 'race',
       'hispanic_origin', 'sex', 'labor_union_member', 'unemployment_reason',
       'employment_status', 'capital_gains', 'capital_losses',
       'stock_dividends', 'tax_filer_status', 'previous_residence_region',
       'previous_residence_state', 'household_status', 'household_summary',
       'instance_weight', ' migration_code_change_in_msa',
       'migration_code_change_in_reg', 'migration_code_move within reg',
       'live_in_this_house_1_year_ago', 'migration_prev_res_in_sunbelt',
       'num_persons_worked_for_employer', 'family_members_under_18',
       'birth_country_father', 'birth_country_mother', 'birth_country',
       'citizenship', 'self_employed', 'veteran_questionnaire',
       'veteran_benfits', 'weeks_worked_in_year', 'year', 'income']

test_cols = train_cols[:-1]


education_dict = {'Children':0,'Less than 1st grade':1,'1st 2nd 3rd or 4th grade':1,'5th or 6th grade':2,'7th and 8th grade':2,
                  '9th grade':2,'10th grade':3,'11th grade':3,'12th grade no diploma':3,'High school graduate':4,
                  'Some college but no degree':4, 'Associates degree-occup /vocational':5,'Associates degree-academic program':5,
                  'Bachelors degree(BA AB BS)':6, 'Masters degree(MA MS MEng MEd MSW MBA)':7,'Prof school degree (MD DDS DVM LLB JD)':8,'Doctorate degree(PhD EdD)':8}

