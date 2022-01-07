#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Description: Configuration class
"""

from pathlib import Path



class Config:
    """
    A class containing every parameters used in the modules
    """
    project_dir = Path(__file__).resolve().parents[2]
    data = Path(__file__).parents[2] / 'data'
    figures = Path(__file__).parents[2] / "reports/figures"

    labels_map = {
    'BRCA': 'BRCA : Breast invasive carcinoma ',
    'COAD': 'COAD : Colon adenocarcinoma',
    'KIRC': 'KIRC : Kidney renal clear cell carcinoma',
    'LUAD': 'LUAD : Lung adenocarcinoma',
    'PRAD': 'PRAD : Prostate adenocarcinoma'
                }

