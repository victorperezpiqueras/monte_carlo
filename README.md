# Jira Monte Carlo Simulation

This code is a Monte Carlo simulation that analyzes the cycle time and story point relationship of issues in a Jira project. It uses the Jira API to retrieve all issues data and performs calculations on the data to provide a statistical analysis of the project's performance.
## Getting Started

To get started, you'll need to create a .env file in the root directory of this project, containing the following environment variables:

    JIRA_URL: The URL of your Jira instance
    JIRA_USER: Your Jira username
    JIRA_PASSWORD: Your Jira password
    JIRA_PROJECT_CODE: The code of the project you want to analyze

## Prerequisites

This code is written in Python and uses the following libraries:

    os: to load environment variables from the .env file
    pd: to create a DataFrame from the issues data
    matplotlib: to create the Monte Carlo distribution and story points vs time analysis plots

## Running the code

To run the code, simply execute the following command:

```python monte_carlo.py```

The code will use the Jira API to retrieve all issues data from the specified project, calculate the cycle times for each issue, and create a Monte Carlo distribution and story points vs time analysis of the data. The resulting data is also saved to a JSON file in the output directory.
## Monte Carlo Distribution:

Monte Carlo Distribution
## Story Points vs Time Analysis:

Story Points vs Time Analysis
