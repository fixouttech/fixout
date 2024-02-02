import mysql.connector
from mysql.connector import Error

import datetime

class MySQLConnector:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance') or not cls.instance:
          cls.instance = super().__new__(cls)
        return cls.instance
 
    def connect(self):
        try:
            self.connection = mysql.connector.connect(host='localhost',
                                                database='fixout',
                                                user='root')
                                                #password='pynative@#29')
            if self.connection.is_connected():
                db_Info = self.connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)
                return self.connection
                #cursor = self.connection.cursor()
                #cursor.execute("select database();")
                #record = cursor.fetchone()
                #print("You're connected to database: ", record)

        except Error as e:
            print("Error while connecting to MySQL", e)
        #finally:
        #    if self.connection.is_connected():
        #        cursor.close()
        #        self.connection.close()
        #        print("MySQL connection is closed")

    def close(self):
        if self.connection.is_connected():
            self.connection.close()
            print("MySQL connection is closed")

    def checkTokenAndReportID(self, reportName, token):
        cursor = self.connection.cursor(buffered=True)
        cursor.execute('SELECT * FROM tokens WHERE token = %s',(token,))
        tokenVerification = cursor.fetchone()
        cursor.close()
        if tokenVerification is not None:
            print("Token verification OK")
            cursor = self.connection.cursor(buffered=True)
            cursor.execute('SELECT * FROM reports WHERE report_name = %s AND token_fk = %s',(reportName,token,))
            isThereReportAlready = cursor.fetchone()
            cursor.close()
            if isThereReportAlready is None:
                return tokenVerification
            else:
                print("A report with the same name is already register in the database.")
                return None
        return None

    def getEmails(self, token):
        cursor = self.connection.cursor(buffered=True)      
        cursor.execute('SELECT email FROM tokens WHERE token = %s',(token,))
        res = cursor.fetchone()
        cursor.close()
        return res

    def saveExplanations(self, allValues):
        cursor = self.connection.cursor()      
        sql = "INSERT INTO explanations (token_fk, indexf, contrib, original_model, sensitive_feature) VALUES (%s, %s, %s, %s, %s)"
        cursor.executemany(sql, allValues)
        return self.connection.commit()

    def saveCalculatedMetrics(self, allValues):

        cursor = self.connection.cursor() 
        sql = "INSERT INTO calculated_metrics (metric_name_fk, indexf, sensitive_f_name, m_value, threshold_value, original_model, reportid_fk) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        cursor.executemany(sql, allValues)
        return self.connection.commit()

    def saveReport(self, token, reportName, reportID):
        now = datetime.datetime.now()
        cursor = self.connection.cursor() 
        sql = "INSERT INTO reports (token_fk, create_time, report_name, report_id) VALUES (%s, %s, %s, %s)"
        values = (token, now, reportName, reportID)
        cursor.execute(sql, values)
        return self.connection.commit()
      


if __name__ == '__main__':
    connector = MySQLConnector()
    connector2 = MySQLConnector()
    connection = connector.connect()
    connection2 = connector2.connect()
    print(connection)
    print(connection2)
