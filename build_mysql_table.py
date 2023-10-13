import json
import re
import argparse
from tqdm import tqdm
import mysql.connector
import string
import numpy as np
import os
from pyarrow import parquet as pq
from data_process.data_processor import *
from utils.word2vec_util import *

import warnings
warnings.filterwarnings('ignore')


def build_single_table(connection, cursor, table_id, pgTitle, pgEntity, secTitle, caption, headers, cells):
    table_name = 'table_' + str(table_id).replace('-', '_')

    # create table
    comment = pgTitle[:100] + "#|+=" + str(pgEntity) + "#|+=" + secTitle[:100] + "#|+=" + caption[:100]
    comment = str(comment).replace('\\', "\\\\").replace("'", "\\\'").replace('"', "\\\"")
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name}("

    for ci in range(len(headers)):
        column_comment = str(headers[ci])[:50].replace('\\', "\\\\").replace("'", "\\\'").replace('"', "\\\"")
        create_table_sql += f"column{str(ci)} TEXT COMMENT "
        create_table_sql += "\'"
        create_table_sql += column_comment
        create_table_sql += '\'' + ", "

    create_table_sql = create_table_sql.rstrip(', ') + ")"
    create_table_sql += "COMMENT = "
    create_table_sql += '\"'
    create_table_sql += comment
    create_table_sql += '\"'

    cursor.execute(create_table_sql)
    connection.commit()

    # preprocess content data
    table_data = []
    for ci in range(len(headers)):
        col_cell = cells[ci]
        for cj in range(len(col_cell)):
            index = col_cell[cj][0]
            cell_id = col_cell[cj][1][0]
            cell_value = col_cell[cj][1][1]

            row_idx = index[0]
            col_idx = index[1]

            existed_rows = len(table_data)
            for _ in range(row_idx + 1 - existed_rows):
                table_data.append([None for _ in range(len(headers))])

            table_data[row_idx][col_idx] = cell_value

    # check if already inserted
    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor.execute(query)
    row_count = cursor.fetchone()[0]
    if row_count != len(table_data):
        # insert data
        for row_idx in range(len(table_data)):
            insert_sql = f"INSERT INTO {table_name}("
            for col_idx in range(len(table_data[row_idx])):
                insert_sql += f"column{str(col_idx)}" + ", "
            insert_sql = insert_sql.rstrip(', ') + ")"
            insert_sql += f" VALUES ("
        
            for col_idx in range(len(table_data[row_idx])):
                if table_data[row_idx][col_idx] != None:
                    insert_sql += '\"'
                    insert_sql += table_data[row_idx][col_idx][:1000].replace('\\', "\\\\").replace("'", "\\'").replace('"', '\\"')
                    insert_sql += '\"' + ", "
                else:
                    insert_sql += 'null' + ", "

            insert_sql = insert_sql.rstrip(', ') + ")"
            cursor.execute(insert_sql)
        connection.commit()

    # generate histogram
    for ci in range(len(headers)):
        column_name = 'column' + str(ci)
        generate_histogram_sql = f"ANALYZE TABLE {table_name} UPDATE HISTOGRAM ON {column_name} WITH 1024 BUCKETS;"
        cursor.execute(generate_histogram_sql)
        cursor.fetchall()
    connection.commit()


def build_wikitables(data_dir, db_name, connection):
    src_data_path = os.path.join(data_dir, "wikitables_v2/test.table_col_type.json")
    database_name = db_name
    cursor = connection.cursor()

    # create database
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    print(f"Database '{database_name}' created successfully.")

    cursor.execute(f"USE {database_name}")
    print(f"Switched to database '{database_name}'.")

    print(f"Building Wikitables...")
    with open(src_data_path, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        for i in tqdm(range(len(fcc_data)), desc="Processing"):
        # for i in range(len(fcc_data)):
            table_id = fcc_data[i][0]
            pgTitle = fcc_data[i][1]
            pgEntity = fcc_data[i][2]
            secTitle = fcc_data[i][3]
            caption = fcc_data[i][4]
            headers = fcc_data[i][5]
            cells = fcc_data[i][6]
            # annotations = fcc_data[i][7]

            build_single_table(connection, cursor, table_id, pgTitle, pgEntity, secTitle, caption, headers, cells)

    cursor.close()


def is_tag_related_wiki(tag_name, wiki_tag_vecs, word2vec):
    tokens = word2vec.get_tokenized_str(tag_name)
    vec = word2vec.text_2_weighted_vector(tokens)

    for wiki_tag_vec in wiki_tag_vecs:
        dot_product = np.dot(vec, wiki_tag_vec)
        magnitude1 = np.linalg.norm(vec)
        magnitude2 = np.linalg.norm(wiki_tag_vec)
        cosine_similarity = dot_product / (magnitude1 * magnitude2)

        if cosine_similarity > 0.6:
            return True
    return False


def is_table_related_wiki(metadata, table_df, git_tag_rel_flags, wikitable_tag_vecs, word2vec):
    column_types = json.loads(metadata[b'gittables'].decode('utf-8'))
    tag_types = ['dbpedia_syntactic_column_types', 'dbpedia_semantic_column_types', 'schema_syntactic_column_types', 'schema_semantic_column_types']
    git_tags = set()
    for col in table_df.columns:
        for tag_type in tag_types:
            if col in column_types[tag_type]:
                git_tags.add(column_types[tag_type][col]['cleaned_label'].lower())
    
    for git_tag in git_tags:
        if git_tag not in git_tag_rel_flags:
            git_tag_rel_flags[git_tag] = is_tag_related_wiki(git_tag, wikitable_tag_vecs, word2vec)

        if git_tag_rel_flags[git_tag] is True:
            return True
    return False


def build_gittables(data_dir, connection, database_name, dataset_name):

    glove_path = os.path.join(data_dir, "glove/glove.6B.50d.txt")
    word2vec = Word2vecUtil(glove_path)

    wikitable_tag_vecs = []

    with open(os.path.join(data_dir, "wikitables_v2/type_vocab.txt"), 'r') as f:
        for line in f:
            _, tag = line.strip().split('\t')
            tokens = word2vec.get_tokenized_str(tag)
            vec = word2vec.text_2_weighted_vector(tokens)
            wikitable_tag_vecs.append(np.array(vec))
    
    git_tag_rel_flags = {}

    cursor = connection.cursor()

    # create database
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    print(f"Database '{database_name}' created successfully.")

    cursor.execute(f"USE {database_name}")
    print(f"Switched to database '{database_name}'.")

    print(f"Building Gittable {dataset_name}...")
    for root, dirs, files in os.walk(os.path.join(data_dir, f"gittables/{dataset_name}_licensed")):
        for fi in tqdm(range(len(files)), desc="Processing"):
            file_name = files[fi]
            if '.parquet' not in file_name:
                continue

            file_path = os.path.join(root, file_name)
            try:
                table = pq.read_table(file_path)
                metadata = table.schema.metadata
                table_df = table.to_pandas()
            except:
                continue

            if is_table_related_wiki(metadata, table_df, git_tag_rel_flags, wikitable_tag_vecs, word2vec):
                continue
    
            max_col = 150 # due to MySQL table has maximum row size limit
            max_row = 10000
            headers = [c for c in table_df.columns]
            headers = headers[:max_col]

            table_id = fi
            pgTitle = ""
            pgEntity = 0
            secTitle = ''
            caption = file_name.replace('.parquet', '')

            cells = [[] for i in range(len(headers))]
            row_i = 0
            for idx, row in table_df.iterrows():
                for col_j, value in enumerate(row):
                    if col_j >= max_col:
                        continue
                    cells[col_j].append([[row_i, col_j], [0, str(value)]])
                row_i += 1
                if row_i >= max_row:
                    break
            
            build_single_table(connection, cursor, table_id, pgTitle, pgEntity, secTitle, caption, headers, cells)

    cursor.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mysql_host", default=None, type=str, required=True)
    parser.add_argument("--mysql_port", default=3306, type=int, required=False)
    parser.add_argument("--mysql_user", default=None, type=str, required=True)
    parser.add_argument("--mysql_password", default=None, type=str, required=True)
    parser.add_argument("--wikitables_database", default=None, type=str, required=True)
    parser.add_argument("--git_parent_database", default=None, type=str, required=True)
    parser.add_argument("--git_real_time_database", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True)

    args = parser.parse_args()

    connection = mysql.connector.connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
    )
    build_wikitables(args.data_dir, args.wikitables_database, connection)

    build_gittables(args.data_dir, connection, args.git_parent_database, 'parent_tables')

    build_gittables(args.data_dir, connection, args.git_real_time_database, 'real_time_tables')

    connection.close()


if __name__ == "__main__":
    main()

