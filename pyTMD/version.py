#!/usr/bin/env python
u"""
version.py (11/2023)
Gets version number with importlib.metadata
"""
import importlib.metadata

# package metadata
metadata = importlib.metadata.metadata("pyTMD")
# get version
version = metadata["version"]
# append "v" before the version
full_version = f"v{version}"
# get project name
project_name = metadata["Name"]
