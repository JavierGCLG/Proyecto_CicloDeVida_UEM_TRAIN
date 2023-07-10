# app/__init__.py
# -*- coding: utf-8 -*-

import os

# ruta del directorio ráiz del proyecto
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

init_cols = [
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
