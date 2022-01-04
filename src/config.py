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
    project_dir = Path(__file__).resolve().parents[1]
    data = Path(__file__).parents[1] / 'data'
    figures = Path(__file__).parents[1] / "reports/figures"

