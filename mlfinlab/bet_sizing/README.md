# mlfinlab - Bet Sizing

## Introduction
This folder contains classes and functions for sizing bets based on a given investment strategy with given bet side confidence, e.g. the output from a machine learning model. The approaches implemented in this module are based on those described in Chapter 10 of "Advances in Financial Machine Learning" by Marcos López de Prado.

## Contents
1. `__init__.py` - init file for the bet sizing module.
2. `ch10_snippets.py` - Python file containing the code snippets adapted from Chapter 10 of López de Prado's book. The functions in this file have been altered for clarity or to comply with PEP8 guidelines, and also have elaborated docstrings and comments, but otherwise remain unchanged.
3. `bet_sizing.py` - Python file containing the user-level functions for bet sizing. All functionality should be access from functions in this file, and these functions are specified in the __init__.py file.
4. `ef3m.py` - Python file containing an implementation of the EF3M algorithm.
