# Network Anomaly Detection Project

The Network Anomaly Detection Dashboard is an interactive web application aimed to showcase insights and analysis in the data of the traffic of the network that could enable findings during the analysis of activities that are malicious in the systems of networks.

## Run Dashboard Instructions

This guide will help you set up the Dashboard on your machine, from installation to navigating through the interactive website.

### Prerequisites

Before you begin, ensure you have the following installed on your machine:
- Python (3.8 or later)
- pip (Python Package Installer)

### Installation

1. **Clone the GitHub Repository / Download zipped folder**

   Start by cloning the repository containing the IDS Dashboard source code to your local machine / I will provide a zip folder of this project in Blackboard so you don't have to clone for supervisors. Use the following command in your terminal:

   ```
   git clone https://github.com/0mar711/3rd_Year_Project.git
   ```

2. **Download Data Files**

   Download the necessary data files from the provided Google Drive link: [Download Data Files](https://drive.google.com/file/d/1qVM1BYoQpP7zJnMh6rJm9cQ7e9MYld6e/view?usp=sharing)

   After downloading, extract the contents of the zipped folder into the root directory of the cloned repository.

3. **Install Required Packages**

   Before proceeding, we recommend that you install the libraries in a fresh environment using Miniconda to make sure they are compatible and to avoid potential conflicts with existing packages. Using a fresh environment also helps in maintaining a clean workspace where only the necessary libraries for this project are installed.

   After setting up your environment, navigate to the root directory of the project in your terminal. Install the required Python packages using the `requirements.txt` file included in the repository:

   ```
   cd 3rd_Year_Project
   pip install -r requirements.txt
   ```

   ⚠️ **Caution**: If you are using a Conda environment, ensure that you are using the `pip` executable that corresponds to your activated Conda environment. This is to prevent the installation of packages in the wrong Python environment, which is a common issue, especially on Windows where the system-wide `pip` might take precedence. You can verify this by using the `where pip` command (on Windows) or `which pip` command (on Unix-like systems) after activating your environment. If the output does not point to the `pip` inside your Conda environment, refer to the troubleshooting steps provided to correct the `PATH` or use the full path to your environment's `pip` executable.

### Running the Server

To launch the Dashboard, execute the following command in your terminal, ensuring you are still in the project's root directory:

```
python runDashboard.py
```

Upon successful execution, you will see a message indicating the server is running and listening on localhost:8050/custom-hub 

### Navigating the Network Anomaly Detection Dashboard

The Anomaly Detection Network Dashboard is a primary platform that demonstrates extensive analysis of the network traffic data across diverse datasets, done by different machine learning models. The description of the interactive web application follows:

#### Dashboard Overview
After going to `/custom-hub/`, a greeting shall be presented with an overview of the aims and objectives. The subsequent section details on the introduction to the aim and goals of the project: the projects involve the development of effective machine learning models for network anomaly detection and the enhancement of the interpretability of the models.

#### Dataset Specific Analysis
The dashboard features detailed analyses for three significant datasets: UNSW-NB15, CIC-IDS 2017, and inSDN. For each dataset, you can explore:

- **Dataset Description**: A brief overview of the dataset, its significance, and the types of network attacks it includes.
- **Model Analyses**: Direct links to comprehensive analyses for different machine learning models applied to the dataset. These include:
  - Random Forest
  - XGBoost
  - LSTM (Long Short-Term Memory Networks)
  - AutoEncoder
  - Reinforcement Learning (DDQN)

Each analysis provides insights into model performance, SHAP values, and more, offering an in-depth understanding of each model's effectiveness in detecting network anomalies.

#### Interactive Components
The dashboard leverages `explainerdashboard`'s interactive components, such as sliders and predict components, allowing you to customize the analysis. These features enable you to:
- Filter data based on specific criteria.
- Adjust model parameters in real-time to observe their impact on the model's performance.
- Explore SHAP values to understand feature importance in the decision-making process of the models.

#### Exploring Different Datasets
To navigate to a specific dataset's analysis, use the navigation bar at the top:
- **Home**: Returns you to the dashboard overview at `/custom-hub/`.
- **UNSW-NB15**: Takes you to `/custom-hub/unsw`, where you can explore analyses related to the UNSW-NB15 dataset.
- **CIC-IDS 2017**: Directs you to `/custom-hub/cic`, focusing on the CIC-IDS 2017 dataset.
- **inSDN**: Leads you to `/custom-hub/insdn`, dedicated to the inSDN dataset analysis.

Each dataset page contains a description of the dataset, a graphical representation of attack distribution (if applicable), and buttons linking to detailed model analyses.

**IMPORTANT NOTE:** The full analysis and componenets is only provided on the random forest and XGBoost models due to the infeasibility and no support for calculating SHAP values for the deep Keras models (LSTM and Reinforcement Learning DDQN)
## Setting Up and Running the Notebooks for Testing

Before diving into the notebooks, you need to download the datasets used in this project from their respective websites.

 **Download Datasets**: Obtain the datasets from the following sources:
   - UNSW-NB15: [Link to dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
   - USB-IDS: [Link to dataset](https://idsdata.ding.unisannio.it/)
   - CIC-IDS 2017: [Link to dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
   - inSDN: [Link to dataset](https://aseados.ucd.ie/datasets/SDN/)

Organize Dataset Files: Create a new folder named datasets in the root directory of the project (3rd_Year_Project). Place all the downloaded dataset files into this datasets folder. This organization is important because the notebooks are configured to load datasets from this specific location.

The project includes detailed Jupyter notebooks for data exploration, preprocessing, model training, and evaluation. Follow the next steps to set up your environment and explore the notebooks:

1. **Set Up the Conda Environment**

   Navigate to the project directory and create a new Conda environment using the provided `environment.yml` file. This file contains all the necessary packages to run the notebook.

   ```
   cd 3rd_Year_Project
   conda env create -f environment.yml
   ```

2. **Activate the Environment**

   Once the environment is ready, activate it:

   ```
   conda activate TensorFlowGPU
   ```

3. **Launching Jupyter Notebook**

   Navigate to the directory containing the notebooks and start Jupyter notebook :

   ```
   jupyter notebook
   ```

### Notebook Structure and Execution Guidelines

Each notebook is structured to guide you through the analysis process for different datasets, including UNSW-NB15, USB-IDS, CIC-IDS 2017, and inSDN. Here's what to expect:

- **Table of Contents**: Easily navigate to different sections, including EDA (Exploratory Data Analysis), preprocessing, and model testing.
- **Preprocessing Steps**: Follow and run all preprocessing steps before model evaluation to ensure data is properly prepared.
- **Model Testing**: Detailed steps for training and evaluating models. To test or evaluate models, execute all preprocessing steps as well as train/test splits within the notebook.

To ensure accurate results, run the notebooks in order, executing each code cell sequentially. You can also experiment with the code to explore different modeling techniques or data insights.

## Acknowledgments

A heartfelt thank you to:

- The open-source community and the creators of `explainerdashboard`, whose tools have been instrumental in this project.
- Researchers and organizations for providing the IDS datasets, enabling the analyses conducted here.
- Everyone dedicated to enhancing network security, whose efforts inspire this work.

Your contributions have made this project possible.

---

