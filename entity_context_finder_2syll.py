"""
Traces list of entities in a given corpus and records the context in which they're found.

Saves to .csv file with existing columns, context_X column values filled up to context_5.

Rough Approach:

1) Load the Dataset: Read your CSV containing entities.
2) Process Each Text File: For each .txt file, search for entities from the dataset.
3) Extract Context: When an entity is found in the text, extract the surrounding text between two 'shads' (།) to capture the context.
4) Store Results: Limit to a maximum of 5 contexts per entity, appending each to the corresponding row in the dataset.
5) Save to New CSV: Save the updated dataset with new context columns.


Saves the order in which the entities are finalised - will offer clues as to which ones need quality-checking -
especially shorter tokens with multiple meanings.

If context is just the place with a shad or something like that, don't save as context.
E.g. "མགྲོན་བསུ་་ཁང" --> "མགྲོན་བསུ་་ཁང་།" and "ལོ་སི" --> "ལོ་སི།"

___

Saves to json in following format:

[
    ["SHAD-SEPARATED PHRASE"=str,
        {"entities":
            [
                [span_start=int,
                span_end=int,
                entity_tag=str
                ]
            ]
        }
    ]
]
"""
import numpy as np
import pandas as pd
import re
import glob
import os
from tqdm import tqdm
import json
from botok import WordTokenizer
from botok.config import Config
from pathlib import Path
from botok import Text

desired_num_contexts= 2  # per entity

### Botok Setup ###


base_dir = '/BOTOK/BASE_DIR'

config = Config(dialect_name="custom", base_path=Path(base_dir))

wt = WordTokenizer(config=config)

####################

def save_checkpoint(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_checkpoint(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def extract_context_around_token(text, text_tokens_list, entity, max_contexts=5):
    """
    Pattern to match: [optional to account for text-beginning] shad or 'ga ' -->  0+ spaces --> 0+ any character -->
    entity --> 0+ any character --> entity --> 0+ any character --> shad or 'ga '
    :param text: [str] text within which you wish to search for the given entity and its surrounding context
    (shad-segmented phrase containing the given entity)
    :param entity: [str] string whose contexts you are searching for within the text provided.
    :param max_contexts: [int] maximum number of contextual examples you wish to extract for the given entity
    :return: [list] containing max_contents number of entities in context (str). Returns list of strings.
    """
    results = [] # list of tuples - each tuple contains (preceding_token, entity, following_token)
    for i, token in enumerate(text_tokens_list):
        if token == entity:  # Exact match
            prev_token = text_tokens_list[i - 1] if i > 0 else None  # Previous token, None if first
            next_token = text_tokens_list[i + 1] if i < len(text_tokens_list) - 1 else None  # Next token, None if last
            results.append((prev_token, entity, next_token))

    # Pattern to match the entity and capture text between the nearest shads around it
    # ?: is non-capturing GROUP
    # [^།]* is 0+ characters excluding shad
    # followed by shad
    # ? is 0 or 1 of non-capturing group (accounts for entity matches in first sentence of text)
    # re.escape() is the entity escaped e.g. with \ in case of non-alphanumeric characters in entity
    # [^།]* is 0+ characters excluding shad
    # ?: is non-capturing group
    # shad
    
    matches = []
    for ent_match in results:
        preceding_entity, entity, following_entity = ent_match
        
        context_pattern = re.compile(rf'(?:(ག |།)\s*)?([^།]*)({re.escape(preceding_entity)})( \s)*({re.escape(entity)})(\s*)({re.escape(following_entity)}ག$\s|{re.escape(following_entity)}[^།]*ག\s|{re.escape(following_entity)}[^།]*།)')

        match = context_pattern.search(text)
        try:
            matches.append(match.group(0))
        except:
            print("error - probably NoneType")

    # Return up to `max_contexts` unique context strings
    unique_matches = []
    for match in matches:
        if match not in unique_matches:
            unique_matches.append(match)
        if len(unique_matches) == max_contexts:
            break
    
    unique_matches = ["".join(match[1:]) for match in unique_matches]

    return unique_matches

def extract_context(text, entity, max_contexts=5):
    """
    Pattern to match: [optional to account for text-beginning] shad or 'ga ' -->  0+ spaces --> 0+ any character -->
    entity --> 0+ any character --> entity --> 0+ any character --> shad or 'ga '
    :param text: [str] text within which you wish to search for the given entity and its surrounding context
    (shad-segmented phrase containing the given entity)
    :param entity: [str] string whose contexts you are searching for within the text provided.
    :param max_contexts: [int] maximum number of contextual examples you wish to extract for the given entity
    :return: [list] containing max_contents number of entities in context (str). Returns list of strings.
    """
    # Pattern to match the entity and capture text between the nearest shads around it
    # ?: is non-capturing GROUP
    # [^།]* is 0+ characters excluding shad
    # followed by shad
    # ? is 0 or 1 of non-capturing group (accounts for entity matches in first sentence of text)
    # re.escape() is the entity escaped e.g. with \ in case of non-alphanumeric characters in entity
    # [^།]* is 0+ characters excluding shad
    # ?: is non-capturing group
    # shad
    context_pattern = re.compile(rf'(?:(ག |།)\s*)?([^།]*)({re.escape(entity)}[^།]*)(ག |།)')
    matches = context_pattern.findall(text)

    # Return up to `max_contexts` unique context strings
    unique_matches = []
    for match in matches:
        if match not in unique_matches:
            unique_matches.append(match)
        if len(unique_matches) == max_contexts:
            break

    unique_matches = ["".join(match[1:]) for match in unique_matches]

    return unique_matches


def count_num_contexts(entity_dict, entity):
    """
    Counts the number of contextual examples you have already extracted for a given entity. Counts non-nan values.
    :param entity_dict: (dict) The entity dictionary whereby key = entity, value is dict
    (key = context_1...5, val = context (str or np.nan)).
    :param entity: (str) unicode Tibetan entity to search for in text.
    :return: integer count of number of contexts already extracted for a given entity.
    """
    context_count = 0

    # print(f"entity_dict:\n{entity_dict[entity]}")

    for k, v in entity_dict[entity].items():
        try:
            if np.isnan(v):
                pass
        except:
            context_count += 1
    return context_count

def get_tokens(wt, text):
    # Using Botok
    tokens = wt.tokenize(text, split_affixes=False)
    return tokens

def return_botok_tokenlist(in_text):
    # Tokenize the text
    tokens = wt.tokenize(in_text)
    
    # Extract tokenized words
    token_list = [token.text for token in tokens]

    return token_list

def process_text_files(entity_list, text_files_path, output_path):
    """
    Searches for contexts of given list of entities in set of .txt files.
    :param entity_list: [str] Filepath to .csv file containing entities for which context is required
    :param text_files_path: [str] Filepath to directory containing text files in which we will search for entity
    contexts
    :param output_path: [str] Filepath to which finalised json file will be saved
    """
    # Load the dataset
    df = pd.read_csv(entity_list)

    # List all columns desired
    context_columns = []
    for i in range(1, desired_num_contexts+1):  # 1 to 5 if desired_num_contexts = 5
        context_columns.append(f"context_{i}")

    sources_list = []
    for i in range(1, desired_num_contexts+1):
        sources_list.append(f"source_{i}")

    # Create dict (entity_dict) in which key = entity, val = dict (key = context_X, value = context str if already
    # in source .csv) OR load if available
    entity_dict = load_checkpoint('entity_dict_2syll.json') or {}

    # Create dict (finalised_entities) in which to transfer entries to once desired_num_contexts (default=5) have
    # been found in texts OR load if available
    finalised_entities = load_checkpoint('finalised_entities_2syll.json') or {}

    # Create dict (context_sources) recording sources for context_1 to context_5 (listed in context_columns).
    # OR load if available
    # key = entity,
    # val = dict(
    # key = 'source_1'...'source_5' (1 for each col):
    # val =  filename of source text in which corresponding entity was found
    # )
    context_sources = load_checkpoint('context_sources_2syll.json') or {}

    # Create list or load checkpoint to record order of entity finalisation (reaching desired_num_contexts for entity)
    # List enables you to quality-check entries likely to contain errors as their context lists were quickly filled
    # (e.g. single character words).
    finalisation_order = load_checkpoint('finalisation_order_2syll.json') or []

    for _, row in df.iterrows():
        unicode_bo_value = row['unicode_bo']

        # Initialise any entities in CSV file if not already processed and stored in loaded checkpoint
        if unicode_bo_value not in entity_dict and unicode_bo_value not in finalised_entities:
            context_dict = {col: row[col] for col in context_columns}
            entity_dict[unicode_bo_value] = context_dict

            source_dict = {source: None for source in sources_list}
            context_sources[unicode_bo_value] = source_dict

    # Process each text file
    text_files = glob.glob(text_files_path + '/**/*.txt', recursive=True)
    # Get the total count
    total_files = len(text_files)

    for filepath in tqdm(text_files, total=total_files, desc=f"Searching for context"): 
        file_name = os.path.basename(filepath)
        print(f"\nSearching file: {file_name}")

        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()

        text_tokens = return_botok_tokenlist(text)

        # For each entity, try to find contexts within the text
        for index, row in df.iterrows():
            entity = row['unicode_bo']

            if entity in finalised_entities:
                # print(f"\nEntity '{entity}' has already been finalised. Skipping.")
                #Skip entity search - no more needed
                continue

            current_num_contexts = count_num_contexts(entity_dict, entity)
            contexts_still_needed = desired_num_contexts - current_num_contexts

            # When reading in dataframe (or loading checkpoints) to initialise dicts, move over anything FULL to
            # finalised dict.
            if current_num_contexts == desired_num_contexts:
                finalised_entities[entity] = entity_dict.pop(entity)

            elif current_num_contexts < desired_num_contexts:
                # Get contexts for the current entity - return a couple of extra contexts than required in case
                # of duplicated contexts between unique contexts returned and contexts previously found.
                if entity in text_tokens:
                    try:
                        new_contexts_found = extract_context_around_token(text, text_tokens, entity, desired_num_contexts+2)
    
                        current_contexts = list(entity_dict[entity].values())
                        contexts_to_add = []
    
                        # Ensure no duplicated contexts
                        for context in new_contexts_found:
                            if contexts_still_needed > 0:
                                if context not in current_contexts:
                                    # Ensure no unhelpful contexts added (just entity + shad or entity + tsheg and shad)
                                    if context not in [entity, f"{entity}་", f"{entity}།", f"{entity}་།"]:
                                        contexts_to_add.append(context)
                                        contexts_still_needed -= 1
    
                        # Update the entity_dict with contexts and source_dict with source in which context was found
                        for context in contexts_to_add:
                            entity_dict[entity][f"context_{current_num_contexts+1}"] = context
                            current_num_contexts += 1
                            context_sources[entity][f"source_{current_num_contexts+1}"] = file_name
    
                        # Transfer entity from entity_dict to finalised_entities once
                        # current_num_context = desired_num_contexts to avoid iterating through unnecessary entities
                        if current_num_contexts == desired_num_contexts:
                            finalised_entities[entity] = entity_dict.pop(entity)
                            finalisation_order.append(entity)  # Track finalisation order
                            save_checkpoint(finalisation_order, 'finalisation_order_2syll.json')  # Save progress
                    except:
                        print(f"error searching for: {entity} - skipping")

        # Save progress after processing each file
        save_checkpoint(entity_dict, 'entity_dict_2syll.json')
        save_checkpoint(finalised_entities, 'finalised_entities_2syll.json')
        save_checkpoint(context_sources, 'context_sources_2syll.json')
        save_checkpoint(finalisation_order, 'finalisation_order_2syll.json')
        print("\nProgress checkpointed - see json files")

    # Save final output to specified path
    with open(output_path, 'w', encoding="utf-8-sig") as f:
        json.dump({
            "finalised_entities": finalised_entities,
            "remaining_entity_dict": entity_dict,
            "context_sources": context_sources,
            "finalisation_order": finalisation_order
        }, f, ensure_ascii=False, indent=4)

    return finalised_entities, entity_dict, context_sources, finalisation_order

def write_contexts_to_df(original_df, output_df_path, completed_entity_list, context_sources):
    """
    :param original_df: Filepath to the original CSV file
    :param output_df_path: Filepath to which finalised .csv file will be saved (same structure as original, with 5
    context columns filled per entity (incl. any additional contexts beyond 5 which were included in the original file)
    :param completed_entity_list: (DICT) Read in from finalised JSON file then passed to function.
    {"entity": {"context_1": "context", "context_2": "context", etc.}}
    :param entity_dict: Dictionary of entities still being processed
    :param context_sources: Dictionary of sources for contexts
    :return:
    """
    # Load the original dataset
    df = pd.read_csv(original_df)

    # Iterate through the list of entities and update the dataframe
    for key, contexts_dict in tqdm(completed_entity_list.items(), desc="Updating dataframe",
                                   total=len(completed_entity_list)):
        if key in df['unicode_bo'].values:
            # Locate the row with the matching unicode_bo
            idx = df[df['unicode_bo'] == key].index[0]
            # Iterate over the context entries
            for context_num_header, context_text in contexts_dict.items():
                # Check if the column exists in the dataframe
                if context_num_header in df.columns:
                    # Update the dataframe at the specific row and column
                    df.at[idx, context_num_header] = context_text
                else:
                    print(f"Column {context_num_header} not found in dataframe.")
            # Iterate over context sources
            for source_num_header, source_text in context_sources[key].items():
                # Check if the column exists in dataframe
                if source_num_header in df.columns:
                    # Ensure the column is of type object (string-compatible) before assignment
                    if df[source_num_header].dtype != 'object':
                        df[source_num_header] = df[source_num_header].astype('object')
                    # Update the dataframe at the specific row and column
                    df.at[idx, source_num_header] = source_text
        else:
            print(f"Key {key} not found in dataframe.")

    # Save the updated dataframe
    df.to_csv(output_df_path, index=False, encoding="utf-8-sig")
    print(f"Processed dataset with contexts saved to {output_df_path}")



    # Saves contexts into existing DF template - cannot write to whole new df because we'll lose contexts beyond
    #  desired_num_contexts. Append source cols to end. Source_cols will cause problems for columns in which you
    #  already had some but less than desired_num_contexts columns filled (from Robbie's dataset). For these, if
    #  num_sources < num_contexts, sources should be right-shifted (i.e. not start from source_1) to align with
    #  corresponding contexts.


# Example usage
entity_list = "PATH/TO/YOUR/DATASET/CSV"  # Replace with your SHORT ENTITY dataset file path
text_files_path = "PATH/TO/YOUR/CORPUS/TXTFILES"  # Folder containing text files
output_final_json_path = "YOUR/CHOSEN/JSON_FILE/OUTPUT/PATH"
output_df_path = "YOUR/CHOSEN/DF/OUTPUT/PATH/CSV"
finalised_entities, entity_dict, context_sources, finalisation_order = process_text_files(entity_list,
                                                                                          text_files_path,
                                                                                          output_final_json_path)

# Load the finalised_entities JSON file
with open("./entity_dict_2syll.json", 'r', encoding='utf-8') as json_file:
    completed_entity_list = json.load(json_file)

with open("./context_sources_2syll.json", 'r', encoding='utf-8') as json_file:
    context_sources = json.load(json_file)
    print(context_sources)

write_contexts_to_df(entity_list, output_df_path, completed_entity_list, context_sources)

