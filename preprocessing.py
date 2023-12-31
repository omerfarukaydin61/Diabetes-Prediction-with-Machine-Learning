import pandas as pd

def replace_zeros_with_median_considering_outcome(data, columns_to_fill=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']):
    def median_target(var):
        temp = data[data[var].notnull()]
        temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
        return temp

    data[columns_to_fill] = data[columns_to_fill].replace(0, pd.NA)

    for column in columns_to_fill:
        median_values = median_target(column)
        for index, row in median_values.iterrows():
            outcome_value = row['Outcome']
            median_value = float(row[column])
            data.loc[(data[column].isna()) & (data['Outcome'] == outcome_value), column] = median_value

    data[columns_to_fill] = data[columns_to_fill].astype(float)

    return data


def replace_zeros_with_median(data, columns_to_fill = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']):

    data[columns_to_fill] = data[columns_to_fill].replace(0, pd.NA)

    data[columns_to_fill] = data[columns_to_fill].apply(lambda x: x.fillna(x.median()))

    return data


