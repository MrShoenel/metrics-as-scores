from sys import path
from pathlib import Path

cli_dir = Path(__file__).resolve().parent
mas_dir = cli_dir.parent
path.append(str(cli_dir))
path.append(str(mas_dir))

from MainWorkflow import MainWorkflow


def cli():
    wf = MainWorkflow()
    wf.print_welcome()

    while not wf.stop:
        wf.main_menu()



if __name__ == '__main__':
    cli()
