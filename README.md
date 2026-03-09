# A practical introduction to machine learning

This repository contains materials for a workshop on machine learning, designed for an audience with no prior experience in the field. The workshop is structured into four parts:
1. [Introduction to Machine Learning](./01_part_Introduction/README.md)
2. [Data Gathering](./02_part_Data_Gathering/README.md)
3. [Data Cleaning & Feature Engineering](./03_part_Data_Cleaning_and_Feature_Engineering/README.md)
4. [Training, Evaluation, and Model Usage](./04_part_Training_Evaluation_and_Model_Usage/README.md)

## Setting up the environment

To follow along with the workshop, you will need to set up a Python environment with the necessary libraries. We recommend using the following approach:
- **Step 0 (only on Windows for the first time)**: Loosening Your Execution Policy
    - Open up Windows Terminal.
    - The execution policy sets how strict your system is about running scripts from other sources. For this tutorial, you'll want to set it to `RemoteSigned`. Run the following command:
        ```powershell
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
        ```

- **Step 1**: Installing Python using UV
    - [Managing Python Projects With uv: An All-in-One Solution](https://realpython.com/python-uv/)
    - The uv tool is a high-speed package and project manager for Python. It’s written in Rust and designed to streamline your workflow. It offers fast dependency installation and integrates various functionalities into a single tool.
    - With uv, you can install and manage multiple Python versions, create virtual environments, efficiently handle project dependencies, reproduce working environments, and even build and publish a project. These capabilities make uv an all-in-one tool for Python project management.
    - Install uv with the official standalone installer:
        - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
        - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - **Upgrading uv**: When uv is installed via the standalone installer, it can update itself on-demand: `uv self update`

- **Step 2**: Installing a specific Python version and creating a virtual environment
    - Install `git` if you haven't already, as it is required to clone the repository. You can download it from [git-scm.com](https://git-scm.com/install).
    - Clone this repository and navigate to the project directory.
        - `cd path\to\your\desired\directory`
        - `git clone https://github.com/ltfe-ict-academy/practical-introduction-to-machine-learning.git`
        - `cd practical-introduction-to-machine-learning`
    - Run the command: `uv python install 3.14.3`
    - Create a new environment with the installed Python version: `uv venv --python 3.14.3`
    - When needed, activate the virtual environment: `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux.
    - Run `python --version` to verify the correct Python version is active.

- **Step 3**: Installing the required libraries
    - With the virtual environment activated, run the following command to install the necessary libraries:
        - `uv pip install -r requirements.txt`

- **Step 4**: Installing and running VS Code
    - [Visual Studio Code](https://code.visualstudio.com/) is a powerful code editor that supports Python development with features like IntelliSense, debugging, and integrated terminal.
    - [Download Visual Studio Code](https://code.visualstudio.com/Download) and install it on your computer.
    - Using a command prompt or terminal navigate into the course folder and open it in VS Code by entering the following commands:
        - `cd path\to\your\desired\directory\practical-introduction-to-machine-learning`
        - `code .`
    - Install the **Python extension for Visual Studio Code** from the [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
        - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
        - [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
        - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
    - Open the Command Palette (`Ctrl+Shift+P`), start typing `Python: Select Interpreter` command from the Command Palette. Select the local Python interpreter installed previously (ex. `(venv-name) Python 3.14.3`).
    - Restart the VS Code to make sure all settings are applied.
