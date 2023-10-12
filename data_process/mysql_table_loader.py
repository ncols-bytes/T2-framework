import mysql.connector
import numpy as np

class MysqlTableLoader():
    def __init__(self, host, port, username, password, database):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.username,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print(f"Connected to MySQL Database({self.database})")
        except mysql.connector.Error as err:
            print(f"Error: {err}")

    def disconnect(self):
        if self.connection is not None and self.connection.is_connected():
            self.connection.close()
            print(f"Disconnected from MySQL Database({self.database})")

    def list_all_tables(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES")
            result = cursor.fetchall()
            cursor.close()
            return result
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None

    def get_histograms(self):
        try:
            cursor = self.connection.cursor()
            get_histogram_sql = f"select * from information_schema.column_statistics where SCHEMA_NAME = '{self.database}'"
            cursor.execute(get_histogram_sql)
            result = cursor.fetchall()
            cursor.close()
            return result
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None

    def get_metadata(self, table_name, histogram_map):
        try:
            cursor = self.connection.cursor()

            table_id = table_name[6:].replace("_", "-")
            
            get_table_comment_sql = f"SELECT table_comment FROM information_schema.tables WHERE table_schema = '{self.database}' AND table_name = '{table_name}'"
            cursor.execute(get_table_comment_sql)
            table_comment = cursor.fetchall()[0]

            metadata_list = str(table_comment[0]).split("#|+=")
            pgTitle = metadata_list[0]
            pgEnt = int(metadata_list[1])
            secTitle = metadata_list[2]
            caption = metadata_list[3]

            get_column_comment_sql = f"SELECT COLUMN_COMMENT FROM information_schema.COLUMNS WHERE table_schema = '{self.database}' AND table_name = '{table_name}'"
            cursor.execute(get_column_comment_sql)
            column_comments = cursor.fetchall()
            headers = []
            for header in column_comments:
                headers.append(header[0])

            histogram = []
            for i in range(len(headers)):
                column_name = "column" + str(i)
                column_histogram = histogram_map[table_name + "||" + column_name]
                histogram.append(column_histogram)
            histogram = np.array(histogram)

            cursor.close()
            return table_id, pgTitle, pgEnt, secTitle, caption, headers, histogram
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None

    
    def get_entity_data(self, table_name, col_num, entity_id_map):
        try:
            cursor = self.connection.cursor()
            table_id = table_name[6:].replace("_", "-")

            select_data_sql = f"select * from {table_name} limit 200"
            cursor.execute(select_data_sql)
            rows = cursor.fetchall()

            entities = [[] for _ in range(col_num)]
            row_idx = 0
            for row in rows:
                col_idx = 0
                for cell_value in row:
                    if cell_value != None:
                        if table_id in entity_id_map:
                            entity_id = entity_id_map[table_id][str(row_idx) + '-' + str(col_idx)]
                        else:
                            entity_id = 0
                        entities[col_idx].append([[row_idx, col_idx], [entity_id, cell_value]])
                    col_idx += 1
                row_idx += 1

            cursor.close()
            return entities
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            return None
