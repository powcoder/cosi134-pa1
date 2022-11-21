https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: Jayeol Chun
# Date: 9/4/20 8:35 PM


def accuracy(classifier, test, verbose=False):
  """quick accuracy calculator

  Args:
    classifier: `trained model`
    test: list of test instances
    verbose: whether to display the % result to terminal or not

  Returns:
    float: accuracy in decimals
  """
  correct = [classifier.classify(x) == x.label for x in test]
  if verbose:
    print("Accuracy: %.2d%% " % (100 * sum(correct) / len(correct)))
  return float(sum(correct)) / len(correct)

def to_level(sense: str, level: int = 2) -> str:
  """converts a sense in string to a desired level

  There are 3 sense levels in PDTB:
    Level 1 senses are the single-word senses like `Temporal` and `Contingency`.
    Level 2 senses add an additional sub-level sense on top of Level 1 senses,
      as in `Expansion.Exception`
    Level 3 senses adds yet another sub-level sense, as in
      `Temporal.Asynchronous.Precedence`.

  This function is used to ensure that all senses do not exceed the desired
  sense level provided as the argument `level`. For example,

  >>> to_level('Expansion.Restatement', level=1)
  'Expansion'
  >>> to_level('Temporal.Asynchronous.Succession', level=2)
  'Temporal.Asynchronous'

  When the input sense has a lower sense level than the desired sense level,
  this function will retain the original sense string. For example,

  >>> to_level('Expansion', level=2)
  'Expansion'
  >>> to_level('Comparison.Contrast', level=3)
  'Comparison.Contrast'

  Args:
    sense (str): a sense as given in a `relaions.json` file
    level (int): a desired sense level

  Returns:
    str: a sense below or at the desired sense level
  """
  s_split = sense.split(".")
  s_join = ".".join(s_split[:level])
  return s_join
