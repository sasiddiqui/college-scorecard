# Sayeed Siddiqui
# Preprocess College Scorecard data into training file
# Operates on https://ed-public-download.apps.cloud.gov/downloads/Most-Recent-Cohorts-All-Data-Elements.csv

import pandas as pd

dat1 = pd.read_csv('../Downloads/Most-Recent-Cohorts-All-Data-Elements.csv')
dat1.drop(dat1.columns[0], axis=1, inplace=True)

# Treat privacy-suppressed as missing values (only do once)
#dat2 = dat1.replace('PrivacySuppressed', '')
#dat2.to_csv('../Downloads/Most-Recent-Cohorts-All-Data-Elements.csv')

# Select only numeric variables
dat2 = dat1.select_dtypes(include=['float64', 'int64'])

# Drop columns which are mostly null or zero
nulls = dat2.isnull().sum()
zeros = dat2.eq(0).sum()
dat3 = dat2.loc[:, nulls + zeros < len(dat2)/2]

# Coalesce related attributes into one
def rep(lst):
	# First sort reverse-alphabetically to get highest year number
	lst.sort(reverse=True)
	# Next sort by string length to get simplest
	lst.sort(key=lambda x: len(x))
	return lst[0]

attrs = list(dat3.columns)
groups = {}
for attr in attrs:
	prefix = attr[0:7] 
	if prefix in groups:
		groups[prefix].append(attr)
	else:
		groups[prefix] = [attr]
attrs = list(map(rep, groups.values()))
dat4 = dat3[attrs]

# Normalize data
dat5 = dat4.fillna(dat4.mean())
dat5 = (dat5 - dat5.mean())/dat5.std()

# Specify excluded columns
excluded = ['UNITID', 'OPEID', 'OPEID6', 'SCH_DEG', 'MAIN', 'CONTROL', 'ST_FIPS', 'REGION', 'LOCALE', 'CCBASIC', 'CCUGPROF', 'CCSIZSET', 'RELAFFIL', 'CURROPER', 'POOLYRS']

# Build list of dependent variables 
dependent = []
tokens = ['CDR', 'RPY', 'DEBT', 'REPAY', 'EARN', 'GT']
# This function tests whether an attribute should be dependent
dep = lambda a: any(map(lambda x: x in a, tokens))
dependent = [attr for attr in attrs if dep(attr)]
excluded.extend(dependent)

Y = dat5['DEBT_MDN']
dat6 = dat5.drop(excluded, axis=1)

# Compute correlations to debt
corrs = abs(dat6.corrwith(Y)).sort_values(ascending=False)

Y.to_csv('data/Y.csv', index=False)
dat6.to_csv('data/processed.csv', index=False)
