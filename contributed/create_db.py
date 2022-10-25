from mysql.connector import connect, Error
#import mysql.connector
##########create database###################
mydb = connect(
  host="localhost",
  user="root",
  password="Prix@123#"
)

mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE Face_attendance_test2")