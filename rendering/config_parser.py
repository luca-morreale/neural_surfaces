
import json
from pathlib import Path

def parse_configuration(file):
    ## load config
    with open(file) as json_file:
        config_text = json.load(json_file)

    configs = []

    ## if there is a reference to file, load config from there
    for i, el in enumerate(config_text):
        if type(el) == str:
            # replace filename
            json_folder = Path(file).parent

            with open(str(json_folder) + '/' + el) as json_file:
                replace_config = json.load(json_file)
                configs.extend(replace_config)
        else:
            configs.append(el)

    return configs