from typing import Any, TypeVar
from rich.console import Console
import questionary
from questionary import Choice


T = TypeVar('T')


class Workflow:
    def __init__(self) -> None:
        self.c = Console()
        self.q = questionary
        self.style_mas = 'bold fg:#8B8000 bg:#000'
        self.style_err = 'bold fg:#8B0000 bg:#000'
    
    def _print_doc(self, more: str='') -> None:
        self.q.print(10*'-' + '\n')
        self.q.print(type(self).__doc__.strip() + more + '\n')
        self.q.print(10*'-' + '\n')
    
    def _wait_for(self, what_for: str) -> None:
        self.q.print(text=f'Press [Enter] {what_for}.', style=self.style_mas, end='')
        return self.q.text(message='', qmark='').ask()
    
    def ask(self, options: list[str], prompt: str='You now have the following options:', rtype: T=int) -> T:
        res = self.askt(
            options=list([(options[idx], idx) for idx in range(len(options))]),
            prompt=prompt)
        if res == None:
            # When user cancels
            return None
        if rtype == str:
            return options[res]
        return res
    
    def askt(self, options: list[tuple[str, T]], prompt: str='You now have the following options:') -> T:
        choices = list([Choice(title=options[idx][0], value=options[idx][1]) for idx in range(len(options))])
        return self.q.select(message=prompt, choices=choices, use_shortcuts=True).ask()
    
    def print_info(self, text_normal: str, text_vital: str=None, end: str=None, arrow: str=' -> ') -> None:
        kwargs = {}
        if end != None:
            kwargs['end'] = end
        
        text_normal = f'{arrow}{text_normal}'
        
        if text_vital == None:
            self.q.print(text=text_normal, **kwargs)
        else:
            self.q.print(text=text_normal, end='')
            self.q.print(text=text_vital, style=self.style_mas, **kwargs)
