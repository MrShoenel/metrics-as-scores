"""
This module contains the workflow for running the interactive web application
locally.
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore, Thread
from typing import Literal
from bokeh.command.util import build_single_handler_application
from bokeh.server.server import Server
from pathlib import Path
from metrics_as_scores.__init__ import IS_MAS_LOCAL
from metrics_as_scores.cli.Workflow import Workflow
from metrics_as_scores.cli.helpers import PathStatus, isint, get_local_datasets, required_files_folders_local_dataset, validate_local_dataset_files
from metrics_as_scores.distribution.distribution import LocalDataset
from questionary import Choice
from traceback import TracebackException
from subprocess import Popen, PIPE, STDOUT
from os import environ
from io import TextIOWrapper
from time import sleep

this_dir = Path(__file__).resolve().parent
webapp_dir = this_dir.parent.joinpath('./webapp')
root_dir = this_dir.parent.parent.parent






class LocalWebserverWorkflow(Workflow):
    __doc__ = '''
This workflow allows you to locally run the interactive web application of
Metrics As Scores, using one of the locally available datasets.
'''
    def __init__(self) -> None:
        super().__init__()
        self.preload: bool = None
        self.port: int = None
        self.use_ds: LocalDataset = None
    

    def _port_and_dataset(self) -> tuple[int, LocalDataset]:
        class LocalDatasetWithMissingFiles(LocalDataset):
            missing_files: list[Path]
        
        def check_port(p: str) -> bool:
            if not isint(p):
                return False
            r = int(p)
            return r > 0 and r < 65536
        
        def check_files(ds: LocalDatasetWithMissingFiles) -> LocalDatasetWithMissingFiles:
            ds['missing_files'] = []
            _, files = validate_local_dataset_files(*required_files_folders_local_dataset(local_ds_id=ds['id']))

            for f in files:
                if 'densities_' in f.name and not files[f] == PathStatus.OK:
                    ds['missing_files'].append(f)
            
            return ds
        

        port = int(self.q.text(message='Enter a Port number:', default=f'{5678}', validate=check_port).ask())
        ds_list = list([
            check_files(ds) for ds in get_local_datasets()
        ])
        ds_list_missing = list([ds for ds in ds_list if len(ds['missing_files']) > 0])
        if len(ds_list_missing) > 0:
            for ds in ds_list_missing:
                self.print_info(text_normal='The following dataset misses pre-generated densities: ', text_vital=f'{ds["name"]} [{ds["id"]}]', arrow='')
                self.print_info(text_normal='The following densities are missing: ', text_vital=', '.join(list([f.name for f in ds['missing_files']])))
        
        ds_list_OK = list([ds for ds in ds_list if len(ds['missing_files']) == 0])
        use_ds: LocalDataset = None
        if len(ds_list_OK) == 0:
            raise Exception('There are no datasets available or none of the datasets is complete.')
        else:
            use_ds = self.q.select(
                message='Please choose one of the following Datasets:',
                choices=list([Choice(title=f"{ds['name']} [{ds['id']}]", value=ds) for ds in ds_list_OK])).ask()
        
        return port, use_ds
    
    def _type_quit_to_exit(self, internal: bool=False) -> None:
        self.q.text(message=f'\n{"Close the browser and type e" if internal else "E"}nter "q" to shut down the application:', validate=lambda q: q.lower().startswith('q')).ask()
    
    def start_server_process(self) -> None:
        class State():
            def __init__(self, running: bool) -> None:
                self.running = running
                self.should_stop: bool = False

        proc: Popen = None
        state = State(running=False)
        success_semaphore = Semaphore(value=0)
        try:
            args = [
                'bokeh', 'serve',
                str(webapp_dir),
                '--port', f'{self.port}',
                '--show',
                '--args', f'dataset={self.use_ds["id"]}',
            ]
            if self.preload:
                args.append('preload')
            
            my_env = { **environ, 'PYTHONUNBUFFERED': '1' }
            proc = Popen(args=args, cwd=str(root_dir.resolve()), stdout=PIPE, stderr=STDOUT, env=my_env)
            stdout_and_err = TextIOWrapper(buffer=open(file=proc.stdout.fileno(), mode='rb', buffering=0), write_through=True, encoding='utf-8', errors='ignore')
            def read1(proc: Popen):
                sleep(3) # Let's give the process 3 seconds to crash.
                while not state.should_stop and proc.returncode is None and not stdout_and_err.closed:
                    line = stdout_and_err.readline().strip()
                    if line == '':
                        sleep(1)
                        continue # Ignore empty lines
                    self.q.print(text=line)

                    if 'cannot start bokeh server' in line.lower():
                        state.running = False
                        success_semaphore.release()
                        break # join this thread
                    if 'bokeh app running at' in line.lower():
                        state.running = True
                        success_semaphore.release()
                        # no break, keep on reading and piping
            Thread(target=lambda: read1(proc=proc)).start()
            success_semaphore.acquire()
        except Exception as ex:
            self.q.print(f'\nCannot start the webserver: {str(ex)}: {"".join(TracebackException.from_exception(ex).format())}\n')
        finally:
            if state.running and proc.poll() is None:
                self._type_quit_to_exit()
            else:
                self.q.print('\nIt was not possible to start the web application. Please read the error message and server log above. Most often, the port is already in use.', style=self.style_err)
            
            state.should_stop = True
            try:
                proc.kill()
            except:
                pass # don't care.
    
    def start_server_internally(self) -> None:
        """
        Start an embedded Bokeh web server with the Metrics As Scores application.
        This is an experimental feature. Its intended purpose is for development and
        debugging, so use at your own risk.
        The embedded application server may make Metrics As Scores and its text-based
        user-interface unresponsive. Therefore, you may have to manually kill and re-
        start the process.
        """
        kwargs = {
            'generate_session_ids': True,
            'redirect_root': True,
            'use_x_headers': False,
            'secret_key': None,
            'num_procs': 1,
            'host': '127.0.0.1',
            'sign_sessions': False,
            'develop': False,
            'port': self.port,
            'use_index': True
        }
        
        args = [f'dataset={self.use_ds["id"]}']
        if self.preload:
            args.append('preload')
        app = build_single_handler_application(
            path = webapp_dir.resolve(), argv=args)

        
        server = Server({'/webapp': app}, **kwargs)
        tpe = ThreadPoolExecutor(1)
        def start_server():
            try:
                self._type_quit_to_exit(internal=True)
                server.stop()
                server.io_loop.stop()
            except:
                pass
        
        tpe.submit(start_server)
        
        server.start()
        server.io_loop.add_callback(server.show, '/')
        server.io_loop.start()
        tpe.shutdown(wait=True)

    def start_server(self) -> None:
        """Main entry point for this workflow."""
        self._print_doc()

        r: Literal['proc', 'internal'] = 'proc'
        if IS_MAS_LOCAL:
            r = self.askt(prompt='Choose Method of Running the Web-App:', options=[
                ('Use External Process (works usually better)', 'proc'),
                ('Run Embedded Application Server (experimental; use only for debugging)', 'internal')
            ])

        self.preload = self.askt(prompt='Would you like to pre-load the entire dataset into memory?', options=[
            ('No', False),
            ('Yes (Only recommended if you have enough memory)', True)
        ])

        try:
            self.port, self.use_ds = self._port_and_dataset()
        except Exception as ex:
            self.q.print(text=f'\n{str(ex)}\n')
            return

        if r == 'proc':
            return self.start_server_process()
        else:
            self.q.print('\nNote that the embedded webserver is an experimental feature that is intended to be used for development and debugging purposes only. It is prone to making the application hang on exit. Therefore, you may have to re-start the process manually.\n', style=self.style_mas)
            return self.start_server_internally()
