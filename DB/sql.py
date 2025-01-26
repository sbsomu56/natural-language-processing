import sqlite3
import pandas as pd
import math
connection=sqlite3.connect("transaction.db")

cursor = connection.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS 
transactions(AccountNo TEXT,DATE TEXT,TRANSACTION_DETAILS TEXT,CHQ_NO INT,VALUE_DATE TEXT, WITHDRAWAL_AMT FLOAT,DEPOSIT_AMT FLOAT,BALANCE_AMT FLOAT)
""")

transaction_df = pd.read_csv('DB/transaction.csv')
transaction_df.columns = ['AccountNo','DATE','TRANSACTION_DETAILS','CHQ_NO','VALUE_DATE','WITHDRAWAL_AMT','DEPOSIT_AMT','BALANCE_AMT']

row = transaction_df.iloc[0]

row['AccountNo'] = int(row['AccountNo'][:-1])

for col_name in ['WITHDRAWAL_AMT','DEPOSIT_AMT','BALANCE_AMT']:
    if not math.isnan(row[col_name]):
        row[col_name] = float(row[col_name].replace(" ","").replace(",",""))




cursor.execute(f"INSERT INTO transactions VALUES ({}, {row['DATE']}, {row['TRANSACTION_DETAILS']}, {row['CHQ_NO']}, {row['VALUE_DATE']}, {float(row['WITHDRAWAL_AMT'].replace(" ","").replace(",",""))}, {float(row['DEPOSIT_AMT'].replace(" ","").replace(",",""))}, {float(row['BALANCE_AMT'].replace(" ","").replace(",",""))} )")