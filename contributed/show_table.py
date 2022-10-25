#import mysql.connector as mysql
import psycopg2

db = psycopg2.connect(
database="face_db", user='postgres', password='facegen@123', host='localhost', port= '5432'
)
# db = mysql.connect(
#     host = "localhost",
#     user = "root",
#     passwd = "Prix@123#",
#     database = "Face_attendance_test2" ###change db name
# )
DB_table_name = "test3"
cursor = db.cursor()

# getting all the tables which are present in 'datacamp' database
# cursor.execute("SHOW TABLES")

# tables = cursor.fetchall() ## it returns list of tables present in the database

# ## showing all the tables one by one
# for table in tables:
#     print(table)

## 'DESC table_name' is used to get all columns information
cursor.execute('DESC test3')

## it will print all the columns as 'tuples' in a list
print(cursor.fetchall())

######insert the data###########
# Name = "Guttappa Sajjan"
# Date = "15-10-2022"
# time = "5"
# Insert_data = "INSERT INTO " + DB_table_name + " (NAME,DATE,TIME) VALUES (%s, %s,%s)"
# VALUES = (str(Name), str(Date), str(time))
# try:
#     cursor.execute(Insert_data, VALUES)
# except Exception as e:
#     print(e)


#cursor = db.cursor()
##########fetch the data#############
# defining the Query
# query = "SELECT * FROM test2"

# ## getting records from the table
# cursor.execute(query)

# ## fetching all records from the 'cursor' object
# records = cursor.fetchall()
# #print(records)
# ## Showing the data
# for record in records:
#     print(record)
############################
## 'DESC table_name' is used to get all columns information
# cursor.execute("DESC test2")

# ## it will print all the columns as 'tuples' in a list
# print(cursor.fetchall())