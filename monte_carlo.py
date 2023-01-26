import os
import requests
import json
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from dotenv import load_dotenv


class MonteCarloSimulation:

    def __init__(self, jira_url, jira_username, jira_token, project_code, show_plots=False):

        self.jira_url = jira_url
        self.jira_username = jira_username
        self.jira_token = jira_token
        self.project_code = project_code
        self.show_plots = show_plots

    def _get_filtered_issues(self, params):
        issues = []
        moreResults = True
        while moreResults:
            response = requests.get(f"{self.jira_url}/search", auth=(self.jira_username, self.jira_token),
                                    params=params)
            # Check if the request was successful
            if response.status_code != 200:
                raise Exception(
                    "Error: Unable to retrieve issues from Jira API")
            # Parse the JSON data from the response
            data = json.loads(response.text)
            # Add the issues to the list
            issues.extend(data["issues"])
            # Check if there are more results to retrieve
            if data["startAt"] + len(issues) < data["total"]:
                params["startAt"] = data["startAt"] + len(data["issues"])
            else:
                moreResults = False

        print(f"Retrieved {len(issues)} issues")
        return issues

    def _load_issues_history(self, issues):
        # Define the Jira API query parameters
        params = {
            "expand": "changelog"
        }
        for issue in issues:
            # Send the GET request to the Jira API and store the response in a variable
            response = requests.get(f"{issue['self']}", auth=(
                self.jira_username, self.jira_token), params=params)

            # Check if the request was successful
            if response.status_code != 200:
                print(
                    f"Error: Unable to retrieve history of changes for issue {issue}")
                continue

            # Parse the JSON data from the response
            changelog = json.loads(response.text)
            # Iterate through the history of changes for the issue
            in_progress_time = None
            done_time = None
            todo_done_time = None
            for history in changelog["changelog"]["histories"]:
                # Iterate through the items in each history
                for item in history["items"]:
                    # Check if the item is a status change
                    if item["field"] == "status":
                        # Check if the status changed to 'In Progress'
                        if item["fromString"] == "To Do" and item["toString"] == "In Progress":
                            in_progress_time = history["created"]
                        # Check if the status changed to 'Done'
                        elif item["toString"] == "Done":
                            done_time = history["created"]
                        if item["fromString"] == "To Do" and item["toString"] == "Done":
                            todo_done_time = history["created"]
            issue["fields"]["in_progress_time"] = in_progress_time
            issue["fields"]["done_time"] = done_time
            issue["fields"]["todo_done_time"] = todo_done_time
            print(".", end="", flush=True)

        print("\nLoaded history of changes for all issues")

    def get_all_issues_data(self):

        # Get the current date and calculate the start and end date for last year
        now = datetime.datetime.now()
        last_year_start = (now - datetime.timedelta(days=365)
                           ).strftime("%Y-%m-%d")
        last_year_end = now.strftime("%Y-%m-%d")

        # check if file in output, load from it:
        if os.path.isfile('output/cycle_time_issues.json'):
            # load dataframe from file:
            issues_df = pd.read_json('output/cycle_time_issues.json')
            # load into a list of jsons
            issues = issues_df.to_dict('records')
        else:
            story_point_field = [x for x in
                                 json.loads(requests.get(f"{self.jira_url}/field",
                                                         auth=(self.jira_username, self.jira_token)).text) if
                                 "Story Points" in x['name']][0]

            # Define the Jira API query parameters
            params = {
                "jql": f"project = {self.project_code}"
                       f" and created >= {last_year_start}"
                       f" and created <= {last_year_end}"
                       f" and status = Done"
                       f" and (issuetype = Bug  or issuetype = Story or issuetype = Task)"
                       f" and 'Story Points[Number]' is not EMPTY",
                "fields": f"status,created,updated,resolutiondate,{story_point_field['id']}",
                "maxResults": 10  # max limit
            }
            issues = monte_carlo._get_filtered_issues(params)

            # update custom field keys:
            [issue.update({'fields': {**issue['fields'], **{'story_points': issue['fields'][story_point_field['id']]}}})
             for issue in issues]
            [issue['fields'].pop(story_point_field['id']) for issue in issues]

            monte_carlo._load_issues_history(issues)

        return issues

    def calculate_cycle_times(self, issues, filter_not_in_progress=False):
        for issue in issues:
            # Check if the issue has a 'Done' status
            if issue["fields"]["done_time"] and issue["fields"]["in_progress_time"]:
                # Calculate the cycle time for the issue
                cycle_time = datetime.datetime.strptime(issue["fields"]["done_time"],
                                                        "%Y-%m-%dT%H:%M:%S.%f%z") - datetime.datetime.strptime(
                    issue["fields"]["in_progress_time"], "%Y-%m-%dT%H:%M:%S.%f%z")
            elif not filter_not_in_progress:
                if issue["fields"]["todo_done_time"]:
                    cycle_time = datetime.datetime.strptime(issue["fields"]["todo_done_time"],
                                                            "%Y-%m-%dT%H:%M:%S.%f%z") - datetime.datetime.strptime(
                        issue["fields"]["created"], "%Y-%m-%dT%H:%M:%S.%f%z")
                elif issue["fields"]["done_time"]:
                    cycle_time = datetime.datetime.strptime(issue["fields"]["done_time"],
                                                            "%Y-%m-%dT%H:%M:%S.%f%z") - datetime.datetime.strptime(
                        issue["fields"]["created"], "%Y-%m-%dT%H:%M:%S.%f%z")
                else:
                    cycle_time = None
            else:
                cycle_time = None

            issue["fields"]["cycle_time"] = cycle_time
        print("Calculated cycle times for all issues")

    def sample_distribution(self, issues, sample_size=10000):
        plt.figure(figsize=(14, 8))

        cycle_times = [issue["fields"]["cycle_time"].total_seconds() for issue in issues if
                       issue["fields"]["cycle_time"]]
        cycle_times_days = [cycle_times_day /
                            86400 for cycle_times_day in cycle_times]
        real_mean = np.mean(cycle_times_days)
        real_std = np.std(cycle_times_days)

        # Create a normal distribution object and PDF for real data
        dist = stats.norm(real_mean, real_std)
        x = np.linspace(-100, 100, 200)
        y = dist.pdf(x)
        print(
            f"REAL DATA: Mean : {real_mean} and Standard deviation : {real_std}")
        plt.plot(x, y, color='black', linestyle='solid',
                 label='Probability Density Function of real data')

        # Generate samples from the probability distribution generated using the cycle times
        samples = stats.norm.rvs(real_mean, real_std, size=sample_size)
        sampled_mean = np.mean(samples)
        sampled_std = np.std(samples)
        print(
            f"SAMPLES: Mean : {sampled_mean} and Standard deviation : {sampled_std}")
        # Calculate statistics for the samples
        t_stat, p_value = stats.ttest_ind(cycle_times_days, samples)
        print(f"t-statistic: {t_stat}")
        print(f"p-value: {p_value}")
        plt.hist(samples, bins=60, density=True, rwidth=0.9,
                 label=f'Estimated from samples with p-value = {p_value}')

        # Calculate time of completing an issue based on the sample distribution with X% confidence
        confidence = 85
        confidence_sampled = np.percentile(samples, confidence)
        plt.axvline(x=confidence_sampled, color='red', linestyle='dashed', linewidth=1,
                    label=f'{confidence}% Confidence Interval for Sampled Data = {confidence_sampled:.2f} days')

        # Add labels and show the plot
        plt.xlabel('Cycle Time in Days')
        plt.xlim(left=0)
        plt.xticks(np.arange(0, 100, 5))
        plt.ylabel('Probability Density')
        plt.legend()
        plt.title(f"Cycle Time Distribution for {sample_size} Samples")
        if self.show_plots:
            plt.show()

        # save to file
        plt.savefig(f"output/cycle_time_distribution_{sample_size}.png")
        plt.close()

    def get_story_points_relationship(self, issues):
        plt.figure(figsize=(14, 8))
        story_points = [issue["fields"]["story_points"]
                        for issue in issues if issue["fields"]["story_points"]]
        cycle_times = [issue["fields"]["cycle_time"].total_seconds() for issue in issues if
                       issue["fields"]["cycle_time"]]
        cycle_times_days = [cycle_times_day /
                            86400 for cycle_times_day in cycle_times]
        plt.scatter(story_points, cycle_times_days)

        # add a linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            story_points, cycle_times_days)
        line = slope * np.array(story_points) + intercept
        plt.plot(story_points, line, 'r-', label='Regression line')

        plt.xticks([1, 2, 3, 5, 8, 13, 20])
        plt.xlabel('Story Points')
        plt.ylabel('Cycle Time in Days')
        plt.title(
            "Cycle Time vs Story Points (R^2= {:.2f})".format(r_value ** 2))

        if self.show_plots:
            plt.show()

        # save to file
        plt.savefig("output/cycle_time_vs_story_points.png")
        plt.close()


if __name__ == "__main__":
    load_dotenv()  # take environment variables from .env.

    # Jira API endpoint and credentials
    jira_url = os.environ.get("JIRA_URL")
    jira_username = os.environ.get("JIRA_USER")
    jira_token = os.environ.get("jira_token")
    project_code = os.environ.get("JIRA_PROJECT_CODE")

    monte_carlo = MonteCarloSimulation(
        jira_url, jira_username, jira_token, project_code, show_plots=True)

    issues = monte_carlo.get_all_issues_data()

    monte_carlo.calculate_cycle_times(issues, filter_not_in_progress=False)

    cycle_time_issues = [
        issue for issue in issues if issue["fields"]["cycle_time"]]

    df = pd.DataFrame(cycle_time_issues)
    df.to_json("output/cycle_time_issues.json")

    monte_carlo.sample_distribution(cycle_time_issues, sample_size=1000000)

    monte_carlo.get_story_points_relationship(cycle_time_issues)
