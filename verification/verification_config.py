import json

class VerificationConfig():
    def __init__(self, config):
        with open(config, 'r') as file:
            json_data = file.read()
        verif_conf = json.loads(json_data)

        self.tag_2_verifier_type = {}
        if 'dl_model' in verif_conf['type_verifier_tags']:
            for tag in verif_conf['type_verifier_tags']['dl_model']:
                self.tag_2_verifier_type[tag] = 'dl_model'
        if 'dict' in verif_conf['type_verifier_tags']:
            for tag in verif_conf['type_verifier_tags']['dict']:
                self.tag_2_verifier_type[tag] = 'dict'
        if 'regex' in verif_conf['type_verifier_tags']:
            for tag in verif_conf['type_verifier_tags']['regex']:
                self.tag_2_verifier_type[tag] = 'regex'
        if 'rf' in verif_conf['type_verifier_tags']:
            for tag in verif_conf['type_verifier_tags']['rf']:
                self.tag_2_verifier_type[tag] = 'rf'

        self.verifier_type_2_tags = verif_conf['type_verifier_tags']
        
    def get_verifier_type_by_tag(self, tag):
        return self.tag_2_verifier_type[tag]

    def get_tags_by_verifier_type(self, verifier_type):
        if verifier_type in self.verifier_type_2_tags:
            return self.verifier_type_2_tags[verifier_type]
        return []