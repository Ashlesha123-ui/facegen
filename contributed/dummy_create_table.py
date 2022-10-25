import psycopg2

#Establishing the connection
conn = psycopg2.connect(
database="face_db", user='postgres', password='facegen@123', host='localhost', port= '5432'
)
#Creating a cursor object using the cursor() method
cursor = conn.cursor()
DB_table_name = "test3"
#Doping EMPLOYEE table if already exists.
cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")
sql = "CREATE TABLE " + DB_table_name + """
                (
                    NAME VARCHAR(50) NOT NULL,
                    CHECKIN VARCHAR(20) NOT NULL

                        );
                """
#Creating table as per requirement
# sql ='''CREATE TABLE EMPLOYEE(
# FIRST_NAME CHAR(20) NOT NULL,
# LAST_NAME CHAR(20),
# AGE INT,
# SEX CHAR(1),
# INCOME FLOAT
# )'''
cursor.execute(sql)
print("Table created successfully........")
conn.commit()
#Closing the connection
conn.close()