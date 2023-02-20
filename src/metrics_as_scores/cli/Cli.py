"""
This is the main entry point for the command line interface (the text
user interface, TUI) of Metrics As Scores. It provides access to a set
of workflows for handling data and running the Web Application.
"""

from metrics_as_scores.cli.MainWorkflow import MainWorkflow


def cli():
    """
    Main routine for the command line interface. It runs the main menu in
    a never-terminating loop (except for when the user presses Ctr+c).
    """
    wf = MainWorkflow()
    wf.print_welcome()

    while not wf.stop:
        try:
            wf.main_menu()
        except KeyboardInterrupt:
            break
        except Exception as ex:
            wf.q.print(text=f'An error ({type(ex).__name__}) has occurred: {str(ex)}', style=wf.style_err)
            wf.q.print(text='\nRestarting main menu.\n', style=wf.style_err)


if __name__ == '__main__':
    cli()
