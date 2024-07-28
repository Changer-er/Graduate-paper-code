from water quality import df_wq

def condition_func(row):
    condition = (row['Temperature\n?C (Min)'] >= 20) & (row['Temperature\n?C (Max)'] <= 30) & (row['Dissolved Oxygen (mg/L) (Min)'] >= 4) & (row['Dissolved Oxygen (mg/L) (Max)'] <= 8) & (row['pH (Min)'] >= 6) & (row['pH (Max)'] <= 8) & (row['Conductivity (?mhos/cm) (Min)'] >= 150) & (row['Conductivity (?mhos/cm) (Max)'] <= 500) & (row['BOD (mg/L) (Max)'] <= 5) & (row['Nitrate N + Nitrite N(mg/L) (Max)'] <= 5.5) & (row['Fecal Coliform (MPN/100ml) (Max)'] <= 200) & (row['Total Coliform (MPN/100ml) (Max)'] <= 500)
    return condition