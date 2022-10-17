import os

def generate_csv(file_name):
    os.system(f'python xml_parser.py {file_name} -s ','')
