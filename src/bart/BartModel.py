"""
This Module represents a custom implementation of a BART-base model that is fine-tuned to predict entities and
output them in a template based format.
This implementation is based on the original paper that introduced the template based approach
@inproceedings{cui-etal-2021-template,
    title = "Template-Based Named Entity Recognition Using {BART}",
    author = "Cui, Leyang  and
      Wu, Yu  and
      Liu, Jian  and
      Yang, Sen  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.161",
    doi = "10.18653/v1/2021.findings-acl.161",
    pages = "1835--1845",
}
"""

from typing import List, Tuple, Dict
import re
from pandas import DataFrame
from simpletransformers.seq2seq.seq2seq_utils import SimpleSummarizationDataset
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from transformers import BartForConditionalGeneration, BartTokenizer, get_scheduler, BertForTokenClassification, GPT2ForTokenClassification
import torch
from torch.optim import AdamW
from simpletransformers.config.model_args import Seq2SeqArgs
from tqdm.auto import tqdm
from src.util.utils import prepare_dataset_bart, parse_entities, clean_up_predictions


class BartCustomModel:
    def __init__(self, all_classes: List, model_path: str = "facebook/bart-base"):
        super(BartCustomModel, self).__init__()

        # all classes that can be predicted
        self.all_classes: List = all_classes

        self.model_name: str = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model, config and tokenizer from checkpoint
        self.model: BartForConditionalGeneration = self._get_model()
        self.tokenizer: BartTokenizer = self._get_tokenizer()
        self.args = self._load_model_args()
        self.args.model_name = self.model_name
        self.args.use_multiprocessing = False
        self.args.max_length = 150

        # hyper-parameters
        self.train_batch_size: int = 20  # since bigger would lead to cuda-memory-error
        self.validation_batch_size: int = 3
        self.epochs: int = 1
        self.learning_rate = 4e-05
        self.epsilon = 1e-09

    def _get_model(self) -> BartForConditionalGeneration:
        """
        Load the model either from hugging face or from a local path
        :return: BartForConditionalGeneration
        """
        return BartForConditionalGeneration.from_pretrained(self.model_name)

    def _get_tokenizer(self) -> BartTokenizer:
        """
        Load the tokenizer either from huggingface or from local path
        :return: BartTokenizer
        """
        return BartTokenizer.from_pretrained(self.model_name)

    def _load_model_args(self) -> Seq2SeqArgs:
        """
        Load the model args either form huggingface or from local path
        :return: Seq2SeqArgs
        """
        args = Seq2SeqArgs()
        args.load(self.model_name)
        return args

    def _get_inputs_dict(self, batch) -> Dict:
        """
        Send input batch to device and transform it into a dictionary
        :param batch:
        :return:
        """
        device = self.device
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone()
        labels[y[:, 1:] == pad_token_id] = -100
        inputs = {
            "input_ids": source_ids.to(device),
            "attention_mask": source_mask.to(device),
            "decoder_input_ids": y_ids.to(device),
            "labels": labels.to(device),
        }
        return inputs

    def custom_train(self, train_data: DataFrame, eval_data: DataFrame):
        """
        Train the custom Bart Model
        :param train_data: The Training Data
        :param eval_data: The Evaluation Data
        :return: None
        """

        # move model to device
        self.model.to(self.device)

        # generate training and validation samples
        train_dataset: SimpleSummarizationDataset = prepare_dataset_bart(data=train_data, args=self.args,
                                                                         tokenizer=self.tokenizer)
        validation_dataset: SimpleSummarizationDataset = prepare_dataset_bart(data=eval_data, args=self.args,
                                                                              tokenizer=self.tokenizer)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.train_batch_size,
            num_workers=0,
        )

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=self.epsilon)
        num_training_steps = len(train_dataloader) // self.epochs
        scheduler = get_scheduler(
            'linear',
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=0,
        )

        global_step = 0
        training_loss = 0.0
        self.model.zero_grad()

        self.model.train()

        for epoch in range(self.epochs):

            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch + 1} of {self.epochs}",
                disable=False,
                mininterval=0,
                position=0,
                leave=True
            )

            for step, batch in enumerate(batch_iterator):
                inputs = self._get_inputs_dict(batch)
                outputs = self.model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]

                # backwards propagation
                loss.backward()

                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1

                # Show Loss
                batch_iterator.set_description(f"Epochs {epoch + 1}/{self.epochs}. Running Loss: {loss.item():9.4f}")

                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # add training loss
                training_loss += loss.item()

            validation_loss = self.validate_model(val_data=validation_dataset)
            print(f"Validation Loss: {validation_loss}")

        return global_step, training_loss / global_step

    def validate_model(self, val_data: SimpleSummarizationDataset) -> float:
        """
        Evaluate the model on given validation data as Dataframe
        :param val_data: Dataframe containing ["input_text", "target_text"] as columns
        :return: the validation loss

        NOTE: This method is used to validate the model during training
        """
        self.model.to(self.device)

        eval_sampler = SequentialSampler(val_data)
        eval_dataloader = DataLoader(val_data, sampler=eval_sampler, batch_size=self.validation_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluation", position=0, leave=True):
            inputs = self._get_inputs_dict(batch)
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs[0]
                eval_loss += loss.mean().item()

            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps

        return eval_loss

    def predict_single(self, texts: List) -> Tuple[List[str], List[str]]:
        """
        Predicts the Entities templates for a given List of Texts
        :param texts: A list of input texts
        :return: A list of the predicted entities

        NOTE since a sentence can consist multiple entities, once an entity is found, the entity is removed from the
        given sentence and a new prediction is done to get more entities, once a "no entities found" is reached its,
        it will be stopped
        """

        self.model.to(self.device)
        all_entities: List = []
        all_entity_names: List = []
        max_num_entities: int = 10

        # prepare data loader and loading bar
        text_dataloader: DataLoader = DataLoader(texts, batch_size=1)

        for text in tqdm(text_dataloader, desc="Prediction", position=0, leave=True):
            text: str = text[0]
            previous_text: str = ""

            found_entities: List = []
            found_entity_names = []

            while True:
                token_ids = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]

                input_ids = token_ids.to(self.device)

                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=self.args.max_length,
                )[0]

                decoded = self.tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                entity, entity_name = parse_entities(label=decoded, is_gold=False)

                # break condition
                if self.early_stopping_criteria(current_entity_name=entity_name,
                                                found_entities=found_entities,
                                                previous_text=previous_text,
                                                current_text=text,
                                                max_num_entities=max_num_entities
                                                ):
                    if len(found_entities) == 0 and entity_name == "no entities found":
                        found_entities.append(entity)
                        found_entity_names.append(entity_name)
                    break

                else:
                    entity_string = " ".join(entity)
                    previous_text = text
                    match = re.search(re.escape(entity_string), text)
                    if match is None:
                        entity_string = self.find_substring(text, entity_string)

                    text = text.replace(entity_string, "")
                    text = re.sub(' +', ' ', text)  # remove whitespaces

                found_entities.append(entity)
                found_entity_names.append(entity_name)

            cleaned_entities, cleaned_names = clean_up_predictions(entities=found_entities,
                                                                   entity_names=found_entity_names,
                                                                   legit_classes=self.all_classes)
            all_entities.append(cleaned_entities)
            all_entity_names.append(cleaned_names)

        return all_entities, all_entity_names

    @staticmethod
    def early_stopping_criteria(current_entity_name, found_entities, current_text, previous_text,
                                max_num_entities):
        """
        Since BART only computes one entity at a time, there must be early stopping criteria, to prevent the model to
        get stuck in an infinity loop.
        Therefore, different criteria are defined to early stop the prediction of a sentence
        :param max_num_entities:
        :param current_entity_name:
        :param previous_text: The text form the iteration before
        :param current_text: The current text
        :param found_entities:
        :return:
        """

        # if no entities were found or entity name is empty or return is empty or maximum predictions is made
        if current_entity_name == "no entities found" \
                or current_entity_name == '' \
                or current_entity_name == [] \
                or len(found_entities) == max_num_entities:
            return True

        # If no entity was cut out of the text
        if current_text == previous_text:
            return True

        return False

    @staticmethod
    def find_substring(text: str, entity: str) -> str:
        """
        search for ways, that the text is part of the entity string. Different pattern matches are performed to
        find if the entity is somehow in the text
        :param text: the input text
        :param entity: the entity
        :return: the submatch if there is otherwise the entity itself
        """
        if re.search(re.escape(entity.lower()), text):
            match = re.search(re.escape(entity.lower()), text)
            start, end = match.span()
            return text[start: end]

        # remove words from the entity to find matches
        if len(entity.split(" ")) > 1:
            for i in range(len(entity.split(" ")) - 2):
                entity = " ".join(entity.split(" ")[: -1])
                if re.search(re.escape(entity), text):
                    match = re.search(re.escape(entity), text)
                    start, end = match.span()
                    return text[start: end]

        return entity
