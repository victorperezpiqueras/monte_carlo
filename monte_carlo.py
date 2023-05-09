import argparse
import os
import requests
import json
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from dotenv import load_dotenv
import seaborn as sns
from matplotlib.dates import WeekdayLocator, DateFormatter


class MonteCarloSimulation:

    def __init__(self, jira_url, jira_username, jira_password, project_code, show_plots=False):

        self.jira_url = jira_url
        self.jira_username = jira_username
        self.jira_password = jira_password
        self.project_code = project_code
        self.show_plots = show_plots
        self.date_start = None
        self.query_time_start = None
        self.query_time_end = None
        self.file_name = None
        self.regenerate = False

    def _get_filtered_issues(self, params):
        issues = []
        moreResults = True
        while moreResults:
            response = requests.get(f"{self.jira_url}/search", auth=(self.jira_username, self.jira_password),
                                    params=params)
            # Check if the request was successful
            if response.status_code != 200:
                raise Exception("Error: Unable to retrieve issues from Jira API")
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
            response = requests.get(f"{issue['self']}", auth=(self.jira_username, self.jira_password), params=params)

            # Check if the request was successful
            if response.status_code != 200:
                print(f"Error: Unable to retrieve history of changes for issue {issue}")
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

    def get_all_issues_data(self, created_in_last_x_days):
        # Get the current date and calculate the start and end date for last year
        now = datetime.datetime.now()
        self.query_time_start = (now - datetime.timedelta(days=created_in_last_x_days))

        self.query_time_end = now

        # check if file in output, load from it:
        self.file_name = f'output/cycle_time_issues_{self.query_time_start.strftime("%Y-%m-%d")}_' \
                         f'{self.query_time_end.strftime("%Y-%m-%d")}.json'

        if os.path.isfile(self.file_name) and not self.regenerate:
            # load dataframe from file:
            issues_df = pd.read_json(self.file_name)
            # load into a list of jsons
            issues = issues_df.to_dict('records')
        else:
            story_point_field = [x for x in
                                 json.loads(requests.get(f"{self.jira_url}/field",
                                                         auth=(self.jira_username, self.jira_password)).text) if
                                 "Story Points" in x['name']][0]

            # Define the Jira API query parameters
            params = {
                "jql": f"project = {self.project_code}"
                       f" and created >= {self.query_time_start.strftime('%Y-%m-%d')}"
                       f" and created <= {self.query_time_end.strftime('%Y-%m-%d')}"
                # and Sprint="tmMCO Sprint 2023/05" 
                       f" and status = Done"
                       f" and  (issuetype = Story or issuetype = Task)"  # (issuetype = Bug  or issuetype = Story or issuetype = Task)
                       f" and 'Story Points[Number]' is not EMPTY",
                "fields": f"status,created,updated,resolutiondate,{story_point_field['id']}",
                "maxResults": 10  # max limit
            }
            issues = self._get_filtered_issues(params)

            # update custom field keys:
            [issue.update({'fields': {**issue['fields'], **{'story_points': issue['fields'][story_point_field['id']]}}})
             for issue in issues]
            [issue['fields'].pop(story_point_field['id']) for issue in issues]

            self._load_issues_history(issues)

            print(f"Storing issues into file {self.file_name}")
            df = pd.DataFrame(issues)
            df.to_json(monte_carlo.file_name)

        return issues

    def calculate_cycle_times(self, issues, remove_auto_done_tasks=False):
        for issue in issues:
            # Check if the issue has a 'Done' status
            if issue["fields"]["done_time"] and issue["fields"]["in_progress_time"]:
                # Calculate the cycle time for the issue
                cycle_time = datetime.datetime.strptime(issue["fields"]["done_time"],
                                                        "%Y-%m-%dT%H:%M:%S.%f%z") - datetime.datetime.strptime(
                    issue["fields"]["in_progress_time"], "%Y-%m-%dT%H:%M:%S.%f%z")
            elif not remove_auto_done_tasks:
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
        cycle_time_issues = [issue for issue in issues if issue["fields"]["cycle_time"]]
        return issues, cycle_time_issues

    def sample_distribution(self, issues, sample_size=10000):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        cycle_times = [issue["fields"]["cycle_time"].total_seconds() for issue in issues if
                       issue["fields"]["cycle_time"]]
        cycle_times_days = [cycle_times_day / 86400 for cycle_times_day in cycle_times]
        real_mean = np.mean(cycle_times_days)
        real_std = np.std(cycle_times_days)
        dist = stats.norm(real_mean, real_std)
        x = np.linspace(-100, 100, 200)
        y = dist.pdf(x)
        print(f"REAL DATA: Mean : {real_mean} and Standard deviation : {real_std}")
        ax2.plot(x, y, color='black', linestyle='solid', label='Probability Density Function of real data')

        samples = stats.norm.rvs(real_mean, real_std, size=sample_size)
        samples = [sample for sample in samples if sample >= 0]
        sns.histplot(data=samples, binwidth=3, label=f"{sample_size} Samples (Histogram)", ax=ax1)

        confidence = 85
        confidence_sampled = np.percentile(samples, confidence)
        ax2.axvline(x=confidence_sampled, color='red', linestyle='dashed', linewidth=1,
                    label=f"{confidence}% Confidence Interval for Sampled Data = {confidence_sampled:.2f} days")

        plt.xlim(left=0)
        plt.xlabel('Cycle Time in Days')
        plt.xticks(np.arange(0, 100, 5))
        plt.ylabel('Probability Density')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2)
        plt.title(f"Cycle Time Distribution for {sample_size} Samples")
        if self.show_plots:
            plt.show()
        plt.close()

    def cycle_times_scatter_plot(self, issues, filter_cycle_time_greater_than_days=None,
                                 filter_closed_in_x_days_start=None, filter_closed_in_x_days_end=None,
                                 show_labels=False):
        plt.figure(figsize=(10, 10))
        issues = [issue for issue in issues if issue["fields"]["cycle_time"]]

        # get now as utc 0
        now = datetime.datetime.now(datetime.timezone.utc)
        if filter_closed_in_x_days_start is not None:
            # keep issues that were closed before today - X days:
            issues = [issue for issue in issues if
                      datetime.datetime.strptime(issue["fields"]["done_time"], "%Y-%m-%dT%H:%M:%S.%f%z") >=
                      now - datetime.timedelta(days=filter_closed_in_x_days_start)]
        if filter_closed_in_x_days_end is not None:
            # keep issues that were closed after today - Y days:
            issues = [issue for issue in issues if
                      datetime.datetime.strptime(issue["fields"]["done_time"], "%Y-%m-%dT%H:%M:%S.%f%z") <=
                      now - datetime.timedelta(days=filter_closed_in_x_days_end)]

        cycle_times = [issue["fields"]["cycle_time"].total_seconds() for issue in issues]
        cycle_times_days = [cycle_times_day / 86400 for cycle_times_day in cycle_times]

        # get datetimes for done issues:
        date_done_issues = [datetime.datetime.strptime(issue["fields"]["done_time"], "%Y-%m-%dT%H:%M:%S.%f%z")
                            for issue in issues]

        if filter_cycle_time_greater_than_days is not None:
            issues = [issue for issue, cycle_time in zip(issues, cycle_times_days)
                      if cycle_time <= filter_cycle_time_greater_than_days]
            cycle_times_days = [cycle_time for cycle_time in cycle_times_days
                                if cycle_time <= filter_cycle_time_greater_than_days]
            date_done_issues = [date_done_issue for date_done_issue, cycle_time in
                                zip(date_done_issues, cycle_times_days)
                                if cycle_time <= filter_cycle_time_greater_than_days]

        plt.scatter(date_done_issues, cycle_times_days, s=20)

        if show_labels:
            # put issue keys on points:
            for i, txt in enumerate(cycle_times_days):
                plt.annotate(issues[i]["key"], (date_done_issues[i], cycle_times_days[i]))

        percentile = 95
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axhline(y=confidence_percentile, color='green', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")
        percentile = 85
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axhline(y=confidence_percentile, color='orange', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")
        percentile = 50
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axhline(y=confidence_percentile, color='red', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")

        plt.xlabel('Date')
        plt.xticks(rotation=90)
        # display only one date per week in x axis:
        from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
        # plot x labels once per week:
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(WeekdayLocator(byweekday=(TH)))

        plt.ylabel('Cycle Time in Days')
        plt.legend()

        data_start_date = self.query_time_start
        data_end_date = self.query_time_end

        if filter_closed_in_x_days_start is not None:
            data_start_date = self.query_time_end - datetime.timedelta(days=filter_closed_in_x_days_start)

        if filter_closed_in_x_days_end is not None:
            data_end_date = self.query_time_end - datetime.timedelta(days=filter_closed_in_x_days_end)

        # get min and max date of done issues:
        data_max_date = max(date_done_issues) + datetime.timedelta(days=5)
        data_min_date = min(date_done_issues) - datetime.timedelta(days=5)
        plt.xlim(left=data_min_date, right=data_max_date)

        plt.title(
            f"Cycle Time Scatter Plot for {len(cycle_times_days)} completed tasks "
            f"(Estimated Stories/Tasks) from {data_start_date.strftime('%Y-%m-%d')} to"
            f" {data_end_date.strftime('%Y-%m-%d')}")

        plt.savefig(f"output/cycle_times_scatter_plot_{data_start_date.strftime('%Y-%m-%d')}-"
                    f"{data_end_date.strftime('%Y-%m-%d')}.png")
        if self.show_plots:
            plt.show()
        plt.close()

    def cycle_times_distribution_plot(self, issues, filter_cycle_time_greater_than_days=None,
                                      filter_closed_in_x_days_start=None, filter_closed_in_x_days_end=None):
        plt.figure(figsize=(10, 5))

        # get now as utc 0
        now = datetime.datetime.now(datetime.timezone.utc)
        if filter_closed_in_x_days_start is not None:
            # filter those issues that were closed in the last x days:
            issues = [issue for issue in issues if
                      datetime.datetime.strptime(issue["fields"]["done_time"], "%Y-%m-%dT%H:%M:%S.%f%z") >=
                      now - datetime.timedelta(days=filter_closed_in_x_days_start)]
        if filter_closed_in_x_days_end is not None:
            # filter those issues that were closed in the last x days:
            issues = [issue for issue in issues if
                      datetime.datetime.strptime(issue["fields"]["done_time"], "%Y-%m-%dT%H:%M:%S.%f%z") <=
                      now - datetime.timedelta(days=filter_closed_in_x_days_end)]

        cycle_times = [issue["fields"]["cycle_time"].total_seconds() for issue in issues if
                       issue["fields"]["cycle_time"]]
        cycle_times_days = [cycle_times_day / 86400 for cycle_times_day in cycle_times]

        if filter_cycle_time_greater_than_days is not None:
            cycle_times_days = [cycle_time for cycle_time in cycle_times_days
                                if cycle_time <= filter_cycle_time_greater_than_days]

        num_bins = int((max(cycle_times_days) - min(cycle_times_days)) / 2)
        values, bins, bars = plt.hist(cycle_times_days, bins=num_bins, rwidth=0.9,
                                      label=f"Histogram of Task cycle times")
        plt.bar_label(bars, fontsize=10)

        percentile = 95
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axvline(x=confidence_percentile, color='green', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")
        percentile = 85
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axvline(x=confidence_percentile, color='orange', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")
        percentile = 50
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axvline(x=confidence_percentile, color='red', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")

        plt.xlim(left=0, right=filter_cycle_time_greater_than_days)
        plt.xlabel('Cycle Time in Days')
        plt.xticks(np.arange(0, 100, 5))
        plt.ylabel('Number of Tasks')
        plt.legend()

        data_start_date = self.query_time_start
        data_end_date = self.query_time_end

        if filter_closed_in_x_days_start is not None:
            data_start_date = self.query_time_end - datetime.timedelta(days=filter_closed_in_x_days_start)

        if filter_closed_in_x_days_end is not None:
            data_end_date = self.query_time_end - datetime.timedelta(days=filter_closed_in_x_days_end)

        plt.title(
            f"Cycle Time Distribution for {len(cycle_times_days)} completed tasks "
            f"(Estimated Stories/Tasks) from {data_start_date.strftime('%Y-%m-%d')} to"
            f" {data_end_date.strftime('%Y-%m-%d')}")

        plt.savefig(f"output/cycle_times_distribution_plot_{data_start_date.strftime('%Y-%m-%d')}-"
                    f"{data_end_date.strftime('%Y-%m-%d')}.png")
        if self.show_plots:
            plt.show()
        plt.close()

    def _sample_distribution_percentile(self, issues, filter_greater_than_days=None):
        min_done_date = min([issue["fields"]["done_time"] for issue in issues if issue["fields"]["done_time"]])
        min_done_date = datetime.datetime.strptime(min_done_date, "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%d/%m/%Y")
        max_done_date = max([issue["fields"]["done_time"] for issue in issues if issue["fields"]["done_time"]])
        max_done_date = datetime.datetime.strptime(max_done_date, "%Y-%m-%dT%H:%M:%S.%f%z").strftime("%d/%m/%Y")

        plt.figure(figsize=(10, 5))
        cycle_times = [issue["fields"]["cycle_time"].total_seconds() for issue in issues if
                       issue["fields"]["cycle_time"]]
        cycle_times_days = [cycle_times_day / 86400 for cycle_times_day in cycle_times]
        if filter_greater_than_days:
            cycle_times_days = [cycle_time for cycle_time in cycle_times_days if cycle_time <= filter_greater_than_days]

        num_bins = int((max(cycle_times_days) - min(cycle_times_days)) / 2)
        values, bins, bars = plt.hist(cycle_times_days, bins=num_bins, rwidth=0.9,
                                      label=f"Histogram of Task cycle times")
        plt.bar_label(bars, fontsize=10)

        percentile = 95
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axvline(x=confidence_percentile, color='green', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")
        percentile = 85
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axvline(x=confidence_percentile, color='orange', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")
        percentile = 50
        confidence_percentile = np.percentile(cycle_times_days, percentile)
        plt.axvline(x=confidence_percentile, color='red', linestyle='dashed', linewidth=1,
                    label=f"{percentile}% Percentile for Task completion = {confidence_percentile:.2f} days")

        plt.xlim(left=0, right=filter_greater_than_days)
        plt.xlabel('Cycle Time in Days')
        plt.xticks(np.arange(0, 100, 5))
        plt.ylabel('Number of Tasks')
        plt.legend()
        plt.title(
            f"Cycle Time Distribution for {len(cycle_times_days)} completed tasks "
            f"(Estimated Stories/Tasks) from {min_done_date} to {max_done_date}")
        if self.show_plots:
            plt.show()
        plt.close()

    def get_story_points_relationship(self, issues):
        plt.figure(figsize=(14, 8))
        story_points = [issue["fields"]["story_points"] for issue in issues if issue["fields"]["story_points"] and
                        issue["fields"]["cycle_time"]]
        cycle_times = [issue["fields"]["cycle_time"].total_seconds() for issue in issues if
                       issue["fields"]["story_points"] and issue["fields"]["cycle_time"]]
        cycle_times_days = [cycle_times_day / 86400 for cycle_times_day in cycle_times]
        plt.scatter(story_points, cycle_times_days)

        # add a linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(story_points, cycle_times_days)
        line = slope * np.array(story_points) + intercept
        plt.plot(story_points, line, 'r-', label='Regression line')

        plt.xticks([1, 2, 3, 5, 8, 13, 20])
        plt.xlabel('Story Points')
        plt.ylabel('Cycle Time in Days')
        plt.title("Cycle Time vs Story Points (R^2= {:.2f})".format(r_value ** 2))
        plt.savefig("output/cycle_time_vs_story_points.png")
        if self.show_plots:
            plt.show()
        plt.close()


if __name__ == "__main__":
    load_dotenv()  # take environment variables from .env.

    # Jira API endpoint and credentials
    jira_url = os.environ.get("JIRA_URL")
    jira_username = os.environ.get("JIRA_USER")
    jira_password = os.environ.get("JIRA_PASSWORD")
    project_code = os.environ.get("JIRA_PROJECT_CODE")

    # read params from command line
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation for Jira')
    parser.add_argument('--regenerate', '-r', action='store_true',
                        help='Request data to Jira API again')
    parser.add_argument('--cycle_time_greater_than_days', '-ctgtd', type=int, default=None,
                        help='Filter cycle times greater than days')
    parser.add_argument('--pbis-subset', '-p', type=str, default="last_month",
                        help='Filter issues that were closed: [last_month, previous_month, all]')
    parser.add_argument('--remove-auto-done-pbis', '-radp', action='store_true',
                        help='Filter issues that were moved from To Do to Done directly without In Progress')
    parser.add_argument('--created_since_days', '-csd', type=int, default=365,
                        help='Filter issues that were created in the last X days')
    parser.add_argument('--show_scatter', '-ss', action='store_true',
                        help='Show scatter plot for cycle time')
    parser.add_argument('--show_scatter_labels', '-ssl', action='store_true',
                        help='Show issues keys in scatter plot for cycle time')
    parser.add_argument('--show_hist', '-sh', action='store_true',
                        help='Show hist plot for cycle time')
    parser.add_argument('--show_cycle_vs_sps', '-scvs', action='store_true',
                        help='Show cycle time vs story points plot')

    args = parser.parse_args()

    print("-----------------------------------")
    print("Running Monte Carlo Simulation for Jira with the following parameters:")
    print(f"  - Regenerate data from Jira API: {args.regenerate}")
    print(f"  - Filter cycle times greater than days: {args.cycle_time_greater_than_days}")
    print(f"  - Filter issues that were closed: {args.pbis_subset}")
    print(f"  - Filter issues that were moved from To Do to Done"
          f" directly without In Progress: {args.remove_auto_done_pbis}")
    print(f"  - Filter issues that were created in the last X days: {args.created_since_days}")
    print(f"  - Show scatter plot for cycle time: {args.show_scatter}")
    print(f"  - Show scatter plot labels for cycle time: {args.show_scatter_labels}")
    print(f"  - Show hist plot for cycle time: {args.show_hist}")
    print(f"  - Show cycle time vs story points plot: {args.show_cycle_vs_sps}")
    print("-----------------------------------")

    monte_carlo = MonteCarloSimulation(jira_url, jira_username, jira_password, project_code, show_plots=True)
    monte_carlo.regenerate = args.regenerate

    issues = monte_carlo.get_all_issues_data(created_in_last_x_days=args.created_since_days)

    issues, cycle_time_issues = monte_carlo.calculate_cycle_times(issues,
                                                                  remove_auto_done_tasks=args.remove_auto_done_pbis)

    filter_cycle_time_greater_than_days = args.cycle_time_greater_than_days

    if args.pbis_subset == "last_month":
        filter_closed_in_x_days_start, filter_closed_in_x_days_end = 30, 0
    elif args.pbis_subset == "previous_month":
        filter_closed_in_x_days_start, filter_closed_in_x_days_end = 60, 30
    else:
        filter_closed_in_x_days_start, filter_closed_in_x_days_end = None, None

    if args.show_scatter:
        monte_carlo.cycle_times_scatter_plot(issues,
                                             filter_cycle_time_greater_than_days=filter_cycle_time_greater_than_days,
                                             filter_closed_in_x_days_start=filter_closed_in_x_days_start,
                                             filter_closed_in_x_days_end=filter_closed_in_x_days_end,
                                             show_labels=args.show_scatter_labels)
    if args.show_hist:
        monte_carlo.cycle_times_distribution_plot(issues,
                                                  filter_cycle_time_greater_than_days=filter_cycle_time_greater_than_days,
                                                  filter_closed_in_x_days_start=filter_closed_in_x_days_start,
                                                  filter_closed_in_x_days_end=filter_closed_in_x_days_end)

    if args.show_cycle_vs_sps:
        monte_carlo.get_story_points_relationship(cycle_time_issues)
