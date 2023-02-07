from concurrent.futures import ThreadPoolExecutor
from Workflow import Workflow
from bokeh.command.util import build_single_handler_application
from bokeh.server.server import Server
from pathlib import Path
from helpers import isint, get_local_datasets
from metrics_as_scores.distribution.distribution import LocalDataset
from questionary import Choice

this_dir = Path(__file__).resolve().parent
webapp_dir = this_dir.parent.joinpath('./webapp')
root_dir = this_dir.parent.parent.parent

class LocalWebserverWorkflow(Workflow):
    def __init__(self) -> None:
        super().__init__()
    

    def _port_and_dataset(self) -> tuple[int, LocalDataset]:
        def check_port(p: str) -> bool:
            if not isint(p):
                return False
            r = int(p)
            return r > 0 and r < 65536

        return (
            int(self.q.text(message='Enter a Port number:', default=f'{5678}', validate=check_port).ask()),
            self.q.select(
                message='Please choose one of the following Datasets:',
                choices=list([Choice(title=ds['name'], value=ds) for ds in get_local_datasets()])).ask())
    
    def _type_quit_to_exit(self) -> None:
        self.q.text(message='Enter "q" to shut down the application:', validate=lambda q: q.lower().startswith('q')).ask()


    def start_server(self) -> None:
        r = self.ask(prompt='Choose Method of Running the Web-App:', options=[
            'Use External Process (works usually better)',
            'Run Internal Application Server'
        ])

        if r == 0:
            return self.start_server_process()
        else:
            return self.start_server_internally()
    

    def start_server_process(self) -> None:
        port, use_dataset = self._port_and_dataset()

        from subprocess import Popen, DEVNULL
        from sys import executable

        proc: Popen = None
        try:
            bokeh = Path(executable).parent.joinpath('./bokeh')
            proc = Popen(args=[
                str(bokeh), 'serve',
                str(webapp_dir),
                '--port', f'{port}',
                '--show',
                '--args', f'dataset={use_dataset["id"]}',
            ], cwd=str(root_dir.resolve()), stdout=DEVNULL, stderr=DEVNULL)
        finally:
            self._type_quit_to_exit()
            try:
                proc.kill()
            except:
                pass # don't care.
    
    def start_server_internally(self) -> None:
        port, use_dataset = self._port_and_dataset()

        kwargs = {
            'generate_session_ids': True,
            'redirect_root': True,
            'use_x_headers': False,
            'secret_key': None,
            'num_procs': 1,
            'host': '127.0.0.1',
            'sign_sessions': False,
            'develop': False,
            'port': port,
            'use_index': True
        }
        
        app = build_single_handler_application(
            path = webapp_dir.resolve(), argv=[f'dataset={use_dataset["id"]}'])

        
        server = Server({'/webapp': app}, **kwargs)
        tpe = ThreadPoolExecutor(1)
        def start_server():
            try:
                self._type_quit_to_exit()
                server.stop()
                server.io_loop.stop()
                server.io_loop.close(all_fds=True)
            except:
                pass
        
        tpe.submit(start_server)
        
        server.start()
        server.io_loop.add_callback(server.show, '/')
        server.io_loop.start()
        tpe.shutdown(wait=True)
        
        
