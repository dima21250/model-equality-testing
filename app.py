# app.py – Streamlit entry point for Model Equality Testing (local‑file version)

"""Multi‑page Streamlit app.

* This main page lets the user **specify the path** to a local ``.pkl`` file that
  contains the model samples (instead of using the file‑upload widget).
* The heavy pickle is loaded **once** and cached with ``st.experimental_singleton``
  so the downstream test pages don’t reload it on every rerun.

Run locally with:
```
streamlit run app.py
```
"""

import streamlit as st
from dotenv import load_dotenv
import os
import pickle
from pathlib import Path

st.set_page_config(page_title="Model Equality Testing", layout="centered")

st.title("Model Equality Testing")
st.markdown(
    """
    This application runs a variety of statistical tests that compare language‑model
    output distributions.  Provide the **absolute path** to a ``my_llm_data.pkl``
    file that contains a dictionary with the keys ``'samples'`` (model →
    ``CompletionSample``) and ``'prompt_map'`` (prompt → id).

    The sidebar will list the available tests – each test appears as a separate
    page under the *Pages* navigation.
    """
)

# ---------- Load data from a local path ----------

# Load environment variables from .env if present
load_dotenv()

# Helper to update a variable in .env file
def update_env_var(key: str, value: str, env_path: str = ".env"):
    """Update or add a key=value pair in the .env file, preserving other lines."""
    try:
        # Read existing lines
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []
        key_eq = f"{key}="
        new_line = f"{key}={value}\n"
        for i, line in enumerate(lines):
            if line.startswith(key_eq):
                lines[i] = new_line
                break
        else:
            lines.append(new_line)
        # Write back
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        st.error(f"Failed to update .env file: {e}")
# Determine default path from environment
env_path = os.getenv("LAST_PICKLE_PATH")
if env_path and not os.path.isfile(env_path):
    st.warning(f"LAST_PICKLE_PATH points to non‑existent file: {env_path}")
    env_path = ""
# Pre‑populate the text input with the env value if valid
path_input = st.text_input(
    "Path to ``my_llm_data.pkl``",
    value=env_path if env_path else "",
    placeholder="/full/path/to/my_llm_data.pkl"
)

if path_input:
    p = Path(path_input)
    if not p.is_file():
        st.error(f"File not found: {p}")
    else:
        @st.experimental_singleton
        def load_pickle(pickle_path: str):
            """Load the pickle and cache the resulting dictionary.

            The singleton ensures the file is read only once per app session,
            even if the user navigates between pages.
            """
            with open(pickle_path, "rb") as f:
                return pickle.load(f)

        try:
            data = load_pickle(str(p))
            if not isinstance(data, dict) or "samples" not in data:
                st.error("Invalid pickle format – expected a dict with a 'samples' key.")
            else:
                st.session_state["loaded_data"] = data
                st.success("Data loaded successfully!")
                # Update .env with the loaded path
                update_env_var("LAST_PICKLE_PATH", str(p))
                st.write(f"Available models: {list(data['samples'].keys())}")
        except Exception as e:
            st.error(f"Failed to load pickle: {e}")
else:
    st.info("Enter the absolute path to a ``.pkl`` file to enable the test pages.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** The actual tests are available under the *Pages* navigation on the left.")
