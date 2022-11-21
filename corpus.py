https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 8/30/20 8:02 PM
import codecs
import json
import os
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from utils import to_level

SENSES = [
  'Temporal',
  'Temporal.Asynchronous',
  'Temporal.Asynchronous.Precedence',
  'Temporal.Asynchronous.Succession',
  'Temporal.Synchrony',
  'Contingency',
  'Contingency.Cause',
  'Contingency.Cause.Reason',
  'Contingency.Cause.Result',
  'Contingency.Condition',
  'Comparison',
  'Comparison.Contrast',
  'Comparison.Concession',
  'Expansion',
  'Expansion.Conjunction',
  'Expansion.Instantiation',
  'Expansion.Restatement',
  'Expansion.Alternative',
  'Expansion.Alternative.Chosen alternative', # can ignore `alternative`, will be dropped during processing
  'Expansion.Exception',
  'EntRel'
]


##################################### DATA #####################################
class Document(ABC):
  """Base Document"""
  @abstractmethod
  def featurize(self):
    pass

  @abstractmethod
  def featurize_vector(self, vocab=None) -> np.ndarray:
    pass

class Name(Document):
  """A single data instance in Names corpus"""
  def __init__(self,
               data: str,
               label: str=None):
    # raw data and label
    self.data = data
    self.label = label

    # raw features
    self.features = self.featurize() # type: Tuple[str, str]

    # numeric features as numpy ndarray, to be initialized when building corpus
    self.feature_vector = []  # type: np.ndarray

  def featurize(self) -> Tuple[str, str]:
    """converts raw text into a feature

    Returns:
      Tuple[str, str]: first and last letter
    """
    name = self.data
    first, last = name[0], name[-1]
    return first, last

  def featurize_vector(self, vocab=None) -> np.ndarray:
    """TODO: turn the features of this data instance into a feature vector"""
    raise NotImplementedError()

class PDTBRelation(Document):
  """A single PDTB relation data instance"""
  def __init__(self,
               arg1: str,
               arg2: str,
               connective: Optional[str] = None,
               label: str = None):
    # raw data and label
    self.arg1 = arg1
    self.arg2 = arg2
    self.connective = connective # may be empty for implicit relations
    self.label = label

    # raw features
    self.features = self.featurize() # type: Tuple[List[str], List[str], List[str]]

    # numeric features as numpy ndarray
    self.feature_vector = []  # type: np.ndarray

  def featurize(self) -> Tuple[List[str], List[str], List[str]]:
    """converts each arg1, arg2 and connective into features

    Returns:
      Tuple[List[str], List[str], List[str]]: unigram for each of arg1, arg2 and
      connective
    """
    return self.arg1.split(), self.arg2.split(), self.connective.split()

  def featurize_vector(self, vocab=None) -> np.ndarray:
    """TODO: turn the features of this data instance into a feature vector"""
    raise NotImplementedError()

#################################### CORPUS ####################################
class Corpus(ABC):
  """Base Corpus"""
  def __init__(self,
               data_dir: str,
               max_num_data: int = -1,
               shuffle: bool = False):
    """`self.documents` stores all data instances, regardless of the dataset
    split. This is especially helpful when the dataset does not provide the
    pre-defined dataset splits, as in the Names corpus. For PDTB corpus, on the
    other hand, `self.documents` will be empty.

    `shuffle` is only useful for Names Corpus, and should always be set to True
    as the Names data files are stored alphabetically.

    Args:
      data_dir (str): where to load data from
      max_num_data (int): max number of items to load. -1 if no upper limit
      shuffle (bool): whether to shuffle data during dataset split
    """
    self.documents = []

    self.train = []
    self.dev = []
    self.test = []

    self.max_num_data = max_num_data

    # 1. load data
    self.load(data_dir)

    # 1a. dataset split
    self.train, self.dev, self.test = self.split_corpus(shuffle=shuffle)

    # 2. (optional) compile vocabs
    self.vocabs = self.compile_vocab(most_common_n=1000)

    # 3. convert features into feature vectors
    self.featurize_documents(self.vocabs)

  def featurize_documents(self, vocabs=None):
    """TODO: for each data instance, create and cache its feature vector"""
    raise NotImplementedError()

  ############################## ABSTRACT METHODS ##############################
  @abstractmethod
  def compile_vocab(self, most_common_n: int = -1):
    pass

  @abstractmethod
  def load(self, data_dir: str):
    pass

  @abstractmethod
  def split_corpus(self, shuffle: bool = False):
    pass

class NamesCorpus(Corpus):
  """Names Corpus"""
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.labels = ['male', 'female']
    self.num_features = 53 # 52 alphabets + bias

  """A collection of names, labeled by gender. See names/README for
  copyright and license."""
  def compile_vocab(self, most_common_n: int = -1):
    """We don't compile vocab for Names corpus since we know in theory what the
    entire vocab space consists of: 52 upper- and lower-case alphabets
    """
    return

  def load(self, data_dir: str):
    """loads from `names` data directory"""
    for filename in os.listdir(data_dir):
      if not filename.endswith('.txt'): # skip README
        continue

      data_file = os.path.join(data_dir, filename)
      label = os.path.splitext(filename)[0]

      with open(data_file, "r") as file:
        for i, line in enumerate(file):
          data = line.strip()
          document = Name(data, label=label)
          self.documents.append(document)

          if 0 < self.max_num_data < i:
            break

  def split_corpus(self, shuffle: bool = False) -> Tuple[List[Name], List[Name], List[Name]]:
    """splits corpus into train, dev and test

    Args:
      shuffle (bool): whether to shuffle data before split

    Returns:
      Tuple[List[Name], List[Name], List[Name]]: split data
    """
    if shuffle:
      random.shuffle(self.documents)
    train = self.documents[:5000]
    dev = self.documents[5000:6000]
    test = self.documents[6000:]
    return train, dev, test

def load_relations(rel_file: str, sense_level: int = 2) -> List['PDTBRelation']:
  """Loads a single relation json file

  Args:
    rel_file (str):  path to a json to be loaded
    sense_level (int): see `to_level` in `utils.py`

  Returns:
    List['BagOfWords']: loaded data as a list of BagOfWords objects
  """
  documents = []
  with codecs.open(rel_file, encoding='utf-8') as pdtb:
    pdtb_lines = pdtb.readlines()
    for pdtb_line in pdtb_lines:
      rel = json.loads(pdtb_line)

      data = rel['Arg1']['RawText']
      data_1 = rel['Connective']['RawText']
      data_2 = rel['Arg2']['RawText']

      # when there are multiple senses, we will only use the first one
      label = to_level(rel['Sense'][0], level=sense_level)

      document = PDTBRelation(data, data_1, data_2, label=label)
      documents.append(document)

  return documents

class PDTBCorpus(Corpus):
  """Penn Discourse TreeBank Corpus"""
  def __init__(self,
               data_dir: str,
               max_num_data: int = -1,
               sense_level: int = 2,
               shuffle: bool = False):
    self.labels = SENSES
    self.sense_level = sense_level

    super().__init__(data_dir, max_num_data, shuffle)

    num_vocabs = sum(len(x) for x in self.vocabs)
    self.num_features = num_vocabs + 1

  def compile_vocab(self, most_common_n: int = -1):
    """TODO: compile vocabulary from corpus. Using `most_common_n` highly
        recommended to keep the size of vocabulary manageable."""
    raise NotImplementedError()

  def load(self, data_dir: str):
    """loads from `pdtb` data directory"""
    data = {}
    for filename in os.listdir(data_dir):
      if not filename.endswith('.json'):
        continue

      dataset_split = os.path.splitext(filename)[0]
      rel_file = os.path.join(data_dir, filename)
      data[dataset_split] = load_relations(rel_file, self.sense_level) # type: List[PDTBRelation]

    self.train = data['train']
    self.dev = data['dev']
    self.test = data['test']

  def split_corpus(self, shuffle: bool = False) -> Tuple[List[PDTBRelation], List[PDTBRelation], List[PDTBRelation]]:
    """PDTB comes with pre-defined dataset split"""
    return self.train, self.dev, self.test
