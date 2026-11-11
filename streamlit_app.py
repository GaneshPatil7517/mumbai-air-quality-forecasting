"""
Root entry point for Streamlit Community Cloud and cloud deployments.
Runs Mumbai BreatheWatch AI located at app/app.py.
"""
import os
import sys
import runpy

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Execute main application
app_path = os.path.join(ROOT_DIR, "app", "app.py")
runpy.run_path(app_path, run_name="__main__")
