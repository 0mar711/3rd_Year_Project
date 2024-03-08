# Network Anomaly Detection Dashboard

Welcome to the Network Anomaly Detection Dashboard, an interactive web application designed to display insights and analysis on network traffic data, enabling the identification and analysis of malicious activities within network systems.

## Getting Started

This guide will help you set up the Dashboard on your machine, from installation to navigating through the interactive website.

### Prerequisites

Before you begin, ensure you have the following installed on your machine:
- Python (3.8 or later)
- pip (Python Package Installer)

### Installation

1. **Clone the GitHub Repository**

   Start by cloning the repository containing the IDS Dashboard source code to your local machine. Use the following command in your terminal:

   ```
   git clone https://github.com/0mar711/3rd_Year_Project.git
   ```

2. **Download Data Files**

   Download the necessary data files from the provided Google Drive link: [Download Data Files](https://drive.google.com/file/d/1qVM1BYoQpP7zJnMh6rJm9cQ7e9MYld6e/view?usp=sharing)

   After downloading, extract the contents of the zipped folder into the root directory of the cloned repository.

3. **Install Required Packages**

   Navigate to the root directory of the project in your terminal, and install the required Python packages using the `requirements.txt` file included in the repository:

   ```
   cd 3rd_Year_Project
   pip install -r requirements.txt
   ```

### Running the Server

To launch the Dashboard, execute the following command in your terminal, ensuring you are still in the project's root directory:

```
python runDashboard.py
```

Upon successful execution, you will see a message indicating the server is running and listening on localhost:8050/custom-hub 

### Navigating the Network Anomaly Detection Dashboard

Welcome to the Network Anomaly Detection Dashboard, a comprehensive platform designed to analyze network traffic data across various datasets using different machine learning models. Here's how to navigate through the interactive web application:

#### Dashboard Overview
Upon visiting `/custom-hub/`, you're greeted with an introduction to the project, highlighting the aims and objectives. This section provides a foundational understanding of the project's goals, including developing effective machine learning models for network anomaly detection and enhancing the interpretability of these models.

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
The dashboard leverages `explainerdashboard`'s interactive components, such as sliders and dropdowns, allowing you to customize the analysis. These features enable you to:
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

#### Conclusion
The Network Anomaly Detection Dashboard is designed to be an educational and analytical tool for understanding network security through machine learning. By exploring different datasets and models, you gain insights into the complexity of network anomaly detection and the capabilities of various machine learning approaches in addressing these challenges.


## Acknowledgments

A heartfelt thank you to:

- The open-source community and the creators of `explainerdashboard`, whose tools have been instrumental in this project.
- Researchers and organizations for providing the IDS datasets, enabling the analyses conducted here.
- Everyone dedicated to enhancing network security, whose efforts inspire this work.

Your contributions have made this project possible.

---

