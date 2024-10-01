# TBDynamics: A Compartmental Model for simulating Mycobacterium tuberculosis transmission dynamics in Vietnam
## Overview:
TBDynamics is a Python-based modeling tool developed to simulating Mycobacterium tuberculosis (M.tb) transmission in Vietnam . Leveraging the power of the [summer](https://summer2.readthedocs.io/en/latest/), a Python-based framework for the creation and execution of compartmental models (or “state-based”) epidemiological models of infectious disease transmission, TBDynamics offers a comprehensive compartmental model that simulates TB transmission, progression, and control strategies.
## Features:
- **Compartmental Modeling**: Implements a sophisticated model capturing various stages of M.tb infection and TB diseases.
- **Customizable Parameters**: Allows users to tailor model parameters.
- **Data Integration**: Capable of incorporating real-world data for more accurate modelling.
- **Simulation and Analysis**: Offers tools for running simulations over specified time periods and analyzing the outcomes.
## Installation Guide:

1. **Install Conda (if not already installed):**

   If Conda is not installed, follow these steps:

   - Download the latest version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual).
   - Follow the installation instructions for your operating system.
   - After installation, verify by opening a terminal and running:

     ```bash
     conda --version
     ```
2. **Install Visual Studio Code (VSCode)**

   Download and install Visual Studio Code from [here](https://code.visualstudio.com/).

   Follow the installation instructions for your operating system.

   Once installed, open VSCode.
3. **Install Git**

   Git is required for version control and cloning the repository.

   Download and install Git from [here](https://git-scm.com/).

   After installation, verify the installation by running:

   ```bash
   git --version
   ```

4. **Create a Directory for TBDynamics**

   In your terminal or command prompt, create a new directory where you will store the TBDynamics codebase:

   ```bash
   mkdir ~/tbdynamics
   ```
5. **Open VSCode and Set Up the Project**

   - Launch Visual Studio Code.
   - Open the `tbdynamics` directory you just created by navigating to **File** > **Open Folder** in the VSCode menu.
   - Open a terminal in VSCode by pressing ``Ctrl + Shift + ` `` (or ``Cmd + Shift + ` `` on Mac).

6. **Create a Conda Environment with Python 3.10**

   In the terminal, create a new Conda environment with Python 3.10:

   ```bash
   conda create --name tb_env python=3.10
   ```

7. **Activate the Conda Environment**

   After the environment is created, activate it by running:

   ```bash
   conda activate tb_env
   ```
8. **Navigate to the `tbdynamics` Directory**

   In the terminal, navigate to the `tbdynamics` folder where the repository will be cloned:

   ```bash
   cd ~/tbdynamics
   ```

9. **Clone the Git Repository**

   Clone the `tbdynamics` repository into the current directory:

   ```bash
   git clone https://github.com/longbui/tbdynamics.git .
   ```
The `.` at the end ensures the repository is cloned directly into the `tbdynamics` folder.

10. **Install the Project in Editable Mode**

    Once the repository is cloned, install the project and its dependencies using the following command:

    ```bash
    pip install -e .
    ```

    This will install the current project in editable mode, allowing you to make changes to the code without needing to reinstall the package.

