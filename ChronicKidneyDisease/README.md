Variable Name Role Type Demographic Description Units Missing Values
age Feature Integer Age  year yes
bp Feature Integer  blood pressure mm/Hg yes
sg Feature Categorical  specific gravity  yes
al Feature Categorical  albumin  yes
su Feature Categorical  sugar  yes
rbc Feature Binary  red blood cells  yes
pc Feature Binary  pus cell  yes
pcc Feature Binary  pus cell clumps  yes
ba Feature Binary  bacteria  yes
bgr Feature Integer  blood glucose random mgs/dl yes
bu Feature Integer  blood urea mgs/dl yes
sc Feature Continuous  serum creatinine mgs/dl yes
sod Feature Integer  sodium mEq/L yes
pot Feature Continuous  potassium mEq/L yes
hemo Feature Continuous  hemoglobin gms yes
pcv Feature Integer  packed cell volume  yes
wbcc Feature Integer  white blood cell count cells/cmm yes
rbcc Feature Continuous  red blood cell count millions/cmm yes
htn Feature Binary  hypertension  yes
dm Feature Binary  diabetes mellitus  yes
cad Feature Binary  coronary artery disease  yes
appet Feature Binary  appetite  yes
pe Feature Binary  pedal edema  yes
ane Feature Binary  anemia  yes
class Target Binary  ckd or not ckd  no

pip install ucimlrepo
Import the dataset into your code
from ucimlrepo import fetch_ucirepo
  
# fetch dataset

chronic_kidney_disease = fetch_ucirepo(id=336)
  
# data (as pandas dataframes)

X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets
  
# metadata

print(chronic_kidney_disease.metadata)
  
# variable information

print(chronic_kidney_disease.variables)
