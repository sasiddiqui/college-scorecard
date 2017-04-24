# Sayeed Siddiqui
# Preprocess College Scorecard data into training file
# Operates on https://ed-public-download.apps.cloud.gov/downloads/Most-Recent-Cohorts-All-Data-Elements.csv

import pandas as pd

dat1 = pd.read_csv('../Downloads/Most-Recent-Cohorts-All-Data-Elements.csv')

# Treat privacy-suppressed as missing values (only do once)
#dat2 = dat1.replace('PrivacySuppressed', '')
#dat2.to_csv('../Downloads/Most-Recent-Cohorts-All-Data-Elements.csv')

# Select only numeric variables
dat2 = dat1.select_dtypes(include=['int64', 'float64'])

# Drop columns which are mostly null or zero
nulls = dat2.isnull().sum()
zeros = dat2.eq(0).sum()
dat3 = dat2.loc[:, nulls + zeros < len(dat2)/2]



# ['OPEID', 'OPEID6', 
