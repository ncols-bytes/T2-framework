import re
import json
import os


class VerificationConfig():
    def __init__(self, config):
        with open(config, 'r') as file:
            json_data = file.read()
        verif_conf = json.loads(json_data)

        self.tag_2_verifier_type = {}
        for tag in verif_conf['type_verifier_tags']['ml']:
            self.tag_2_verifier_type[tag] = 'ml'
        for tag in verif_conf['type_verifier_tags']['dict']:
            self.tag_2_verifier_type[tag] = 'dict'
        for tag in verif_conf['type_verifier_tags']['regex']:
            self.tag_2_verifier_type[tag] = 'regex'

        self.verifier_type_2_tags = verif_conf['type_verifier_tags']
        
    def get_verifier_type_by_tag(self, tag):
        return self.tag_2_verifier_type[tag]

    def get_tags_by_verifier_type(self, verifier_type):
        return self.verifier_type_2_tags[verifier_type]

class DictVerifiers():
    def __init__(self, data_dir):
        self.tag_2_header_set = {}
        self.tag_2_cell_set = {}
        
        for root, dirs, files in os.walk(data_dir):
            for file_name in files:
                if "table_col_type.json" not in file_name:
                    continue

                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as file:
                    fcc_data = json.load(file)
                    for table_idx in range(len(fcc_data)):
                        headers = fcc_data[table_idx][5]
                        cells = fcc_data[table_idx][6]
                        tags = fcc_data[table_idx][7]

                        for column_idx in range(len(headers)):
                            header = headers[column_idx]

                            column_name_set = set()
                            cell_set = set()

                            column_name_set.add(header)
                            cell_set.add(header)

                            for cell in cells[column_idx]:
                                cell_value = cell[1][1]
                                cell_set.add(cell_value)

                            for tag in tags[column_idx]:
                                if tag not in self.tag_2_cell_set:
                                    self.tag_2_header_set[tag] = set()
                                    self.tag_2_cell_set[tag] = set()
                                self.tag_2_header_set[tag] |= column_name_set
                                self.tag_2_cell_set[tag] |= cell_set
    
    def verify(self, tag, header, cells):
        if header not in self.tag_2_header_set[tag]:
            return False
        
        cell_hit_cnt = 0
        for cell in cells:
            if cell in self.tag_2_cell_set[tag]:
                cell_hit_cnt += 1
        cell_hit_rate = cell_hit_cnt * 1.0 / len(cells)

        if cell_hit_rate < 1.0:
            return False
        return True


class RegexVerifiers():
    def __init__(self):
        self.tag_2_pattern = {
            "astronomy.orbital_relationship" : "(?!19\d{2}|20[0-2]\d)\d{3,5} ([A-Z][a-z\u00C0-\u01E0]{1,15}|[A-Z][a-z\u00C0-\u01E0]{0,8}[-']?[A-Z][a-z\u00C0-\u01E0]{1,8})$",
            "astronomy.star_system_body" : "(?!19\d{2}|20[0-2]\d)\d{3,5} ([A-Z][a-z\u00C0-\u01E0]{1,15}|[A-Z][a-z\u00C0-\u01E0]{0,8}[-']?[A-Z][a-z\u00C0-\u01E0]{1,8})$",
            "astronomy.asteroid" : "(?!19\d{2}|20[0-2]\d)\d{3,5} ([A-Z][a-z\u00C0-\u01E0]{1,15}|[A-Z][a-z\u00C0-\u01E0]{0,8}[-']?[A-Z][a-z\u00C0-\u01E0]{1,8})$",
            "astronomy.astronomical_discovery" : "(?!19\d{2}|20[0-2]\d)\d{3,5} ([A-Z][a-z\u00C0-\u01E0]{1,15}|[A-Z][a-z\u00C0-\u01E0]{0,8}[-']?[A-Z][a-z\u00C0-\u01E0]{1,8})$",
        }

    def verify(self, tag, cells):
        cell_hit_cnt = 0
        for cell in cells:
            if re.match(self.tag_2_pattern[tag], cell):
                cell_hit_cnt += 1
        cell_hit_rate = cell_hit_cnt * 1.0 / len(cells)

        if cell_hit_rate <= 0.4:
            return False
        return True