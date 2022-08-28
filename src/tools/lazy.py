from concurrent.futures import Future
from threading import Semaphore, Thread
from typing import Callable, TypeVar, Generic, Union
from typing_extensions import Self


T = TypeVar('T')


class Lazy(Generic[T]):
    def __init__(self, fnCreateVal: Callable[[], T]) -> None:
        self._val: T = None
        self._has_val = False
        self._semaphore = Semaphore(1)
        self._fnCreateVal = fnCreateVal
    
    def unset_value(self) -> Self:
        try:
            self._semaphore.acquire()
            self._val = None
            self._has_val = False
        finally:
            self._semaphore.release()

        return self

    @property
    def has_value_volatile(self) -> bool:
        return self._has_val

    @property
    def has_value(self) -> bool:
        try:
            self._semaphore.acquire()
            return self._has_val
        finally:
            self._semaphore.release()

    @property
    def value_volatile(self) -> Union[None, T]:
        return self._val

    @property
    def value(self) -> T:
        try:
            self._semaphore.acquire()
            if not self._has_val:
                self._val = self._fnCreateVal()
                self._has_val = True
            return self._val
        finally:
            self._semaphore.release()

    @property
    def value_future(self) -> Future[T]:
        f = Future()

        temp = self._val
        if type(temp) is T and self.has_value_volatile:
            f.set_result(temp)
        else:
            def set_val():
                try:
                    f.set_result(self.value)
                except Exception as e:
                    f.set_exception(e)
            Thread(target=set_val).start()

        return f
