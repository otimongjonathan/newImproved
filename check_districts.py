import pandas as pd

df = pd.read_csv('train_dataset_cleaned.csv')

print('REGIONS:')
print(df['Region'].unique())

print('\nDISTRICTS by REGION:')
for region in df['Region'].unique():
    print(f'\n{region}:')
    districts = sorted(df[df['Region']==region]['District'].unique())
    print(districts)

print('\nCROPS:')
print(sorted(df['Crop'].unique()))
