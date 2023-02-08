"""
This is the main entry point for the command line interface (the text
user interface, TUI) of Metrics As Scores. It provides access to a set
of workflows for handling data and running the Web Application.
"""

from metrics_as_scores.cli.MainWorkflow import MainWorkflow


def cli():
    wf = MainWorkflow()
    wf.print_welcome()

    while not wf.stop:
        wf.main_menu()


if __name__ == '__main__':
    cli()
