import psycopg2
import os
from io import StringIO
import pandas as pd
from sqlalchemy import create_engine

#connect to database
conn = psycopg2.connect("Database info")
cur = conn.cursor()

cur.execute( """
CREATE TABLE Snow(
    Age smallint,
    Amount money,
    Check_Date date,
    DOB date,
    DOH date,
    DOT timestamp,
    Department_Nbr integer,
    Division_Nbr integer,
    ED_Code varchar(3),
    ED_Name character(11),
    Employee_Name character(40),
    Employee_Nbr numeric(3) PRIMARY KEY,
    Hours numeric(5,2),
    Month character(4),
    Position character(32),
    Sex character(1),
    Status character(2),
    Zip numeric(5)
)
""")
conn.commit()
#create dataframe
os.chdir("/Users/JTBras/Dropbox/HRA Team Folder")
df = pd.read_csv("Snow.csv",sep=',')
df['Age'] = df['Age'].astype(int)
#convert missing age values to numerics then to integers
import numpy as np
df['Age'] = np.nan_to_num(df['Age']).astype(int)
df['Age'] = df['Age'].astype(int)


#copy csv data to database
cur = conn.cursor()
with open('Snow.csv', 'r') as f:
    next(f)
    cur.copy_from(f,'Snow',sep=',')

#practice making queries from database
con = psycopg2.connect("database info")
cur = con.cursor()
cur.execute("""SELECT * FROM Snow""")
