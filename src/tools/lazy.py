from concurrent.futures import Future, ThreadPoolExecutor
from threading import Semaphore, Timer
from typing import Any, Callable, TypeVar, Generic, Union
from typing_extensions import Self
from gc import collect


T = TypeVar('T')


class Lazy(Generic[T]):
    def __init__(self, fn_create_val: Callable[[], T]) -> None:
        self._val: T = None
        self._has_val = False
        self._semaphore = Semaphore(1)
        self._fn_create_val = fn_create_val
        self._tpe = ThreadPoolExecutor(max_workers=1)
    
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
                self._val = self._fn_create_val()
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
            self._tpe.submit(set_val)

        return f


class SelfResetLazy(Generic[T]):
    def __init__(self, fn_create_val: Callable[[], T], fn_destroy_val: Callable[[T], Any]=None, reset_after: float=None) -> None:
        self._val: T = None
        self._has_val = False
        self._semaphore = Semaphore(1)

        self._fn_create_val = fn_create_val
        self._fn_destroy_val = fn_destroy_val
        self._reset_after = reset_after
        self._timer: Timer = None
        self._tpe = ThreadPoolExecutor(max_workers=1)
    
    @property
    def reset_after(self):
        try:
            self._semaphore.acquire()
            return self._reset_after
        finally:
            self._semaphore.release()
    
    @reset_after.setter
    def reset_after(self, value: float=None):
        try:
            self._semaphore.acquire()
            self._reset_after = value
            self._set_timer() # Conditionally re-sets a timer
        finally:
            self._semaphore.release()
        
        return self
    
    def unset_value(self):
        try:
            self._semaphore.acquire()
            self._unset_timer()
            if self._has_val:
                if callable(self._fn_destroy_val):
                    self._fn_destroy_val(self._val) # Pass in the current value
                self._val = None
                self._has_val = False
                collect()
        finally:
            self._semaphore.release()

        return self
    
    def _unset_timer(self):
        if type(self._timer) is Timer and self._timer.is_alive():
            self._timer.cancel()
            del self._timer
            self._timer = None
        return self

    def _set_timer(self):
        self._unset_timer()

        if type(self._reset_after) is float and self._reset_after > 0.0:
            self._timer = Timer(interval=self._reset_after, function=self.unset_value)
            self._timer.start()
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

    def value_volatile(self) -> Union[None, T]:
        return self._val

    @property
    def value(self) -> T:
        try:
            self._semaphore.acquire()
            if not self._has_val:
                self._val = self._fn_create_val()
                self._has_val = True
                self._set_timer()
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
            self._tpe.submit(set_val)

        return f
