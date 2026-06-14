
"""Utils for data load, save, and process (e.g., prompt construction)"""

import os
import json
import yaml
import re
import ast

from ..utils import save_json,extract,judge_multi_choice
from ..base_dataset import BaseDataset

from ..question_formats import get_multiple_choice_prompt

DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}


CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """
    
    start_chr = 'A'
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches

def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': None, 'question_type': data['question_type']}
    else:
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': data['image_1'], 'question_type': data['question_type']}


# DATA SAVING
def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)

def save_jsonl(filename, data):
    """
    Save a dictionary of data to a JSON Lines file with the filename as key and caption as value.

    Args:
        filename (str): The path to the file where the data should be saved.
        data (dict): The dictionary containing the data to save where key is the image path and value is the caption.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for img_path, caption in data.items():
            # Extract the base filename without the extension
            base_filename = os.path.basename(img_path)
            # Create a JSON object with the filename as the key and caption as the value
            json_record = json.dumps({base_filename: caption}, ensure_ascii=False)
            # Write the JSON object to the file, one per line
            f.write(json_record + '\n')

def save_args(args, path_dir):
    argsDict = args.__dict__
    with open(path_dir + 'setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

def construct_prompt(sample):
    """
    Constructs a prompt from a sample, refactored for clarity and safety.

    This version adopts cleaner patterns for option generation, similar to the
    provided `llava_prompt` example.
    """
    question = sample['question']
    question_type = sample.get('question_type', 'multiple-choice') # Default to mc
    res_dict = {}

    if question_type == 'multiple-choice':
        # 1. Safely parse the options string into a list
        # Using ast.literal_eval is much safer than eval()
        try:
            options = ast.literal_eval(sample['options'])
            if not isinstance(options, list):
                raise ValueError("Options could not be parsed into a list.")
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse options for question: {question[:50]}... Error: {e}")
            options = [] # Handle cases with invalid options format

        option_letters = [chr(ord('A') + i) for i in range(len(options))]
        formatted_options_list = [
            f"{letter}. {option_text}" for letter, option_text in zip(option_letters, options)
        ]
        
        final_prompt = get_multiple_choice_prompt(
            question, 
            formatted_options_list,  
            os.environ.get("REASONING") == "True"
        )

        # Populate the result dictionary for evaluation and other metadata
        res_dict['index2ans'] = {letter: text for letter, text in zip(option_letters, options)}
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = option_letters
        res_dict['empty_prompt'] = final_prompt # The new structure creates the final prompt directly
        res_dict['final_input_prompt'] = final_prompt
        res_dict['gt_content'] = sample.get('answer') # For mc, the answer is the choice itself
        
        # Find the full text of the ground truth answer
        try:
            gt_index = option_letters.index(sample['answer'].upper())
            res_dict['gt_content'] = options[gt_index]
        except (ValueError, IndexError):
            # Handle cases where the answer key is not in the options
            res_dict['gt_content'] = "Invalid answer key provided"

    else: # Handles open-ended questions
        if os.environ.get("REASONING") == "True":
            instruction = 'Answer the question using a single word or phrase and put the answer in one "\\boxed{}".'
        else:
            instruction = """Answer the question using a single word or phrase."""
        
        # Use a clean f-string to format
        final_prompt = f"{question}\n{instruction}"

        res_dict['empty_prompt'] = final_prompt
        res_dict['final_input_prompt'] = final_prompt
        res_dict['gt_content'] = sample['answer']

    # Add all original sample data to the result dictionary
    res_dict.update(sample)
    return res_dict

