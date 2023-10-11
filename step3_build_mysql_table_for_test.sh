python data_process/build_mysql_table.py \
    --mysql_host=$(mysql_host) \
    --mysql_user=$(mysql_user) \
    --mysql_password=$(mysql_password) \
    --wikitables_database=wikitable \
    --gittables1_database=parent_tables \
    --gittables2_database=real_time_tables \
    --data_dir="./data"
