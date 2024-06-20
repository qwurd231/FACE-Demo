import sqlite3
from datetime import datetime, timedelta

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Exception as e:
        print(e)

def insert_data(conn, insert_data_sql):
    try:
        c = conn.cursor()
        c.execute(insert_data_sql)
        conn.commit()
    except Exception as e:
        print(e)

def select_data(conn, select_data_sql):
    try:
        c = conn.cursor()
        c.execute(select_data_sql)
        rows = c.fetchall()
        return rows
    except Exception as e:
        print(e)

def delete_data(conn, delete_data_sql):
    try:
        c = conn.cursor()
        c.execute(delete_data_sql)
        conn.commit()
    except Exception as e:
        print(e)

def delete_all_data(conn, delete_all_data_sql):
    try:
        c = conn.cursor()
        c.execute(delete_all_data_sql)
        conn.commit()
    except Exception as e:
        print(e)

def delete_old_data(conn, delete_old_data_sql):
    try:
        cursor = conn.cursor()
    
        # Calculate the time 60 seconds ago
        cutoff_time = datetime.now() - timedelta(seconds=60)
        
        # Execute the deletion query
        cursor.execute(delete_old_data_sql, (cutoff_time,))
        conn.commit()
        
        cursor.close()
    except Exception as e:
        print(e)