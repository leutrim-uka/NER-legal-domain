import re
from typing import List, Dict, Tuple

import pandas as pd
from pandas import DataFrame
from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.seq2seq.seq2seq_utils import SimpleSummarizationDataset
from transformers import BartTokenizer


def load_iob_data(p_train: str = None, p_val: str = None, p_test: str = None) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load the train, validation and test data from csv_files
    :param p_train: path to train data
    :param p_val: path to validation data
    :param p_test: path to test data
    :return: (train_data, test_data, validation_data) as data frames

    NOTE: iob must be transformed back to list from string using eval method reset index to avoid index errors
    """

    p_train = p_train if p_train is not None else "./data/iob_train_data.csv"
    p_val = p_val if p_val is not None else "./data/iob_test_data.csv"
    p_test = p_test if p_test is not None else "./data/iob_validation_data.csv"

    # train data
    train_data: DataFrame = pd.read_csv(p_train, index_col=0)
    train_data["iob"] = train_data["iob"].map(lambda x: eval(x))
    train_data["words"] = train_data["words"].map(lambda x: eval(x))
    train_data['word_labels'] = train_data['iob'].apply(lambda x: ','.join(x))

    # validation data
    validation_data: DataFrame = pd.read_csv(p_val, index_col=0)
    validation_data["iob"] = validation_data["iob"].map(lambda x: eval(x))
    validation_data["words"] = validation_data["words"].map(lambda x: eval(x))
    validation_data['word_labels'] = validation_data['iob'].apply(lambda x: ','.join(x))

    # test data
    test_data: DataFrame = pd.read_csv(p_test, index_col=0)
    test_data["iob"] = test_data["iob"].map(lambda x: eval(x))
    test_data["words"] = test_data["words"].map(lambda x: eval(x))
    test_data['word_labels'] = test_data['iob'].apply(lambda x: ','.join(x))

    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    return train_data, validation_data, test_data


def load_bart_data(p_train: str = None, p_validation: str = None, p_test: str = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Load train, validation and test data for the BART mode in the template format
    :param p_train: path to train data
    :param p_validation: path to validation data
    :param p_test: path to test data
    :return: (train_data, validation_data, test_data) as data frames
    """
    path_train = "./data/bart_train_data_single.csv"
    path_dev = "./data/bart_dev_data_single.csv"
    path_test = "./data/bart_test_data.csv"

    p_train = p_train if p_train else path_train
    p_validation = p_validation if p_validation else path_dev
    p_test = p_test if p_test else path_test

    train_data = pd.read_csv(p_train, sep=',').values.tolist()
    train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])

    eval_data = pd.read_csv(p_validation, sep=',').values.tolist()
    eval_df = pd.DataFrame(eval_data, columns=["input_text", "target_text"])

    test_data = pd.read_csv(p_test, sep=',').values.tolist()
    test_df = pd.DataFrame(test_data, columns=["input_text", "entities", "entity names"])
    test_df["entities"] = test_df["entities"].map(lambda x: eval(x))
    test_df["entity names"] = test_df["entity names"].map(lambda x: eval(x))

    return train_df, eval_df, test_df


def prepare_dataset_bart(data: DataFrame, args: Seq2SeqArgs, tokenizer: BartTokenizer):
    """
    Prepare the dataframe as subclass of Torch.utils.Dataset to use for Bart Training
    :param tokenizer: BartTokenizer
    :param data: A dataframe containing ["input_text", "target_text"] as columns
    :param args: an instance of Seq2SeqArgs for Bart Model
    :return: Dataset containing tokenized input
    """

    return SimpleSummarizationDataset(tokenizer, args, data, "train")


def check_overlapping(value: str, to_match: List[str]):
    """
    Check whether the predicted entity is partially overlapping the actual entity or probably inside
    :param value: the predicted entity
    :param to_match: a list of possible matches
    :return: the number of matches

    NOTE:
    e.g.: (true) HDFC Bank | (predicted) HDFC Bank, Fort, Mumbai
    here, the direct match would not work, but HDFC and Bank were found correctly
    """
    overlap_count: int = 0
    inside_count: int = 0

    # check if prediction is inside the entity or overlapping it
    for match in to_match:
        if re.search(re.escape(value), match):
            inside_count += 1
            span = re.search(re.escape(value), match).span()
            return True, len(match[span[0]:span[1]].split(" "))
        elif re.search(re.escape(match), value):
            overlap_count += 1
            span = re.search(re.escape(match), value).span()
            return True, len(value[span[0]:span[1]].split(" "))

    return False, 0


def prepare_gold_dict(keys: List, values: List, remove_whitespace=True) -> Dict:
    """
    Prepare a dictionary with key: entity class and value entity string.
    Whitespaces will be removed to ensure, that no whitespace errors occur where one whitespace in the data produces
    non-matching strings
    :param remove_whitespace:
    :param keys:
    :param values:
    :return:
    """

    gold_dict: Dict = {key: [] for key in keys}
    for key, val in zip(keys, values):
        if remove_whitespace:
            val = re.sub(" +", "", val)
        gold_dict[key].append(val)

    return gold_dict


def parse_entities(label: str, is_gold: bool) -> Tuple[List[str], str]:
    """
    Parse the original entity name and words form the gold label
    :param is_gold: The gold labels start with <s> because the model otherwise cuts off the first letters from the
    labels. Thus, they need to be removed for obtaining the real entity
    :param label: the gold label
    :return: a list of single words of the entity as well as the entity name

    NOTE: The gold labels are all in the following format [<s> <entities words> is a <entity>]
    e.g.:[<s> Constitution is a statute]
    Where 'Constitution' is the entity itself and 'statute' is the entity name.

    All word that belong to the template are ["is", "a", "an"] where an is only used if <entity> is an organization
    """
    label = label.strip()  # remove whitespaces from beginning and end

    # if no entity was found early return
    if label in ["<s> no entities found", "no entities found"]:
        return ["no entities found"], "no entities found"

    if is_gold:  # remove beginning <s>
        parts: List[str] = label.split(" ")[1:]
    else:
        parts: List[str] = label.split(" ")

    entity_name: str = parts[-1]  # take last part of the label which is the entity name

    # since case number is the only entity that consists of multiple words, taking the last part does not work.
    # thus if the last part of the entity name is only name it is converted to case number
    if entity_name.strip() == "number":
        entity_name = "case number"

    # Filter all words that belong to the template
    if entity_name == "case number":
        entity_parts: List[str] = [w for w in parts if w not in ["is", "a", "an", "case", "number"]]

    else:
        entity_parts: List[str] = [w for w in parts if w not in ["is", "a", "an", entity_name]]

    return entity_parts, entity_name


def clean_up_predictions(entities: List, entity_names: List, legit_classes: List) -> Tuple[List, List]:
    """
    Clean Up the entity list from BART prediction to avoid duplicate predictions or predictions that do not belong
    to any of the legit entity types
    :param entities: a list of lists, containing all entities
    :param entity_names: a list of all entity names
    :param legit_classes: a list of all entity names that are possible to predict
    :return: cleaned entities and cleaned entity names
    """
    if len(entity_names) == 1 and entity_names[0] == "no entities found":
        return entities, entity_names

    # check whether an entity was found multiple times
    if len(set([" ".join(e) for e in entities])) != len(entities):
        entities, entity_names = remove_duplicates(entities, entity_names)

    clean_names, clean_entities = [], []

    for e, name in zip(entities, entity_names):
        if name in legit_classes:
            clean_entities.append(e)
            clean_names.append(name)

    assert len(clean_entities) == len(clean_names)

    # if the model only predicted classes that do not exist, return a no entities found
    if len(clean_names) == 0:
        return [['no entities found']], ['no entities found']

    return clean_entities, clean_names


def remove_duplicates(entities, names):
    """
    Take two lists and remove all duplicates from the first list while keeping the indexes of the second list as well
    :param entities: a list of entities
    :param names: a list of names for the entities
    :return: cleaned names and entities
    """

    as_string = [" ".join(e) for e in entities]
    already_seen = set()
    clean_entities, clean_names = [], []

    for idx, e in enumerate(as_string):
        if e in already_seen:
            continue
        else:
            already_seen.add(e)
            clean_entities.append(e.split(" "))
            clean_names.append(names[idx])

    assert len(clean_entities) == len(clean_names)
    return clean_entities, clean_names


def transform_to_iob(texts, entities, names):
    """
    Transform the parsed BART predictions into BIO
    :param texts: original texts
    :param entities: entities
    :param names: names of the entities
    :return: for each text a list of BIO tokens


    NOTE: Since it is crucial that gild labels as well as the predictions have the same length in BIO format, several
    processing steps are performed to ensure correct formatting
    """
    all_iob = []
    all_iob_text = []
    for text, entity, name in zip(texts, entities, names):
        clean_text = re.sub('[^A-Za-z0-9 ()\[\]]+', ' ', text)
        clean_text = re.sub(' +', ' ', clean_text)

        # clean all unnecessary punctuation
        iob_list = clean_text.split(" ")

        for e, n in zip(entity, name):

            # if no entities were found skip
            if e == "no entities found":
                continue

            # clean entity
            clean_entity = re.sub('[^A-Za-z0-9 ()\[\]]+', ' ', e)
            clean_entity = re.sub(' +', ' ', clean_entity)

            # find the exact match in the texts
            match = re.search(re.escape(clean_entity), clean_text)
            if match is None:
                continue
            start, end = match.span()
            submatch = clean_text[start: end]

            # find where the entity is located by counting the whitespaces up to that point
            text_before_entity: str = clean_text[: start]
            num_spaces: int = text_before_entity.count(" ")
            number_entities = len(submatch.split(" "))

            for _ in range(number_entities):
                iob_list[num_spaces] = "I-" + n.replace(" ", "").upper()
                num_spaces += 1

                if num_spaces >= len(iob_list):
                    break

        all_iob_text.append(iob_list)
        all_iob.append(["O" if not token.startswith("I-") else token for token in iob_list])

    return all_iob, all_iob_text
