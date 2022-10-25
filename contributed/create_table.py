from getpass import getpass
#from mysql.connector import connect, Error
import psycopg2
DB_table_name = "test2"

sql = "CREATE TABLE " + DB_table_name + """
                (
                    NAME VARCHAR(50) NOT NULL,
                    CHECKIN VARCHAR(20) NOT NULL

                        );
                """
# try:
#     cursor.execute(sql)  ##for create a table
# except Exception as ex:
#     print(ex)  #                        # PRIMARY KEY (NAME)
# try:
with psycopg2.connect(
    host="localhost",
    database='face_db',             ####change db name 
    user="postgres",
    password="facegen@123",
    port= '5432'
) as connection:
    #create_db_query = "CREATE DATABASE online_movie_rating"
    with connection.cursor() as cursor:
        #cursor.execute(create_db_query)
        cursor.execute(sql)
        # tables = cursor.fetchall() ## it returns list of tables present in the database

        # ## showing all the tables one by one
        # for table in tables:
        #     print(table)
# except Error as e:
#     print(e)