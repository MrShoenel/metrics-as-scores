from concurrent.futures import Future, ThreadPoolExecutor
from threading import Semaphore, Timer
from typing import Any, Callable, TypeVar, Generic, Union
from typing_extensions import Self
from gc import collect


T = TypeVar('T')


class Lazy(Generic[T]):
    """
    Helper class to lazily compute a value, thread-safe.
    """
    def __init__(self, fn_create_val: Callable[[], T]) -> None:
        """
        fn_create_val: ``Callable[[], T]``
            Function that produces the desired value.
        """
        self._val: T = None
        self._has_val = False
        self._semaphore = Semaphore(1)
        self._fn_create_val = fn_create_val
        self._tpe = ThreadPoolExecutor(max_workers=1)
    
    def unset_value(self) -> Self:
        """
        Used to explicitly unset (destroy) the previously produced value.
        This method is thread-safe.
        """
        try:
            self._semaphore.acquire()
            self._val = None
            self._has_val = False
        finally:
            self._semaphore.release()

        return self

    @property
    def has_value_volatile(self) -> bool:
        """
        Checks, in a volatile manner, if a value is present.
        """
        return self._has_val

    @property
    def has_value(self) -> bool:
        """
        Checks whether a value is present, after having acquired the lock.
        Note that the value may be gone after method exit anyway.
        """
        try:
            self._semaphore.acquire()
            return self._has_val
        finally:
            self._semaphore.release()

    @property
    def value_volatile(self) -> Union[None, T]:
        """
        Returns, in a volatile manner, the value. The value may or may not exist.
        """
        return self._val

    @property
    def value(self) -> T:
        """
        Gets the value. If it was not produced previously, it will be produced first.
        Therefore, this getter is guaranteed to return a value (provided the factory
        of the value does not throw).
        """
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
        """
        Gets an awaitable :py:class:`Future` that will yield the produced value
        once it is available.
        """
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
    """
    Similar to :py:class:`Lazy`, this class also automatically destroys its value
    after some timeout, so that subsequent requests to it force the factory to
    produce a new instance.
    """
    def __init__(self, fn_create_val: Callable[[], T], fn_destroy_val: Callable[[T], Any]=None, reset_after: float=None) -> None:
        """
        fn_create_val: ``Callable[[], T]``
            Function that produces the desired value.
        
        fn_destroy_val: ``Callable[[T], Any]``
            Function that will be given the value before it is de-referenced here.
        
        reset_after: ``float``
            Amount of time, in seconds, after which the produced value ought to be destroyed.
        """
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
        """
        Thread-safe getter for the reset-after property.
        """
        try:
            self._semaphore.acquire()
            return self._reset_after
        finally:
            self._semaphore.release()
    
    @reset_after.setter
    def reset_after(self, value: float=None):
        """
        Thread-safe setter for the reset-after property.
        """
        try:
            self._semaphore.acquire()
            self._reset_after = value
            self._set_timer() # Conditionally re-sets a timer
        finally:
            self._semaphore.release()
        
        return self
    
    def unset_value(self):
        """
        Thread-safe method to destroy a previously produced value. If a value is
        present, it is passed to py:meth:`fn_destroy_val()` first.
        """
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
        """
        Checks, in a volatile manner, if a value is present.
        """
        return self._has_val

    @property
    def has_value(self) -> bool:
        """
        Thread-safe getter for checking if a value is present.
        """
        try:
            self._semaphore.acquire()
            return self._has_val
        finally:
            self._semaphore.release()

    def value_volatile(self) -> Union[None, T]:
        """
        Volatile getter for the may-not-present value.
        """
        return self._val

    @property
    def value(self) -> T:
        """
        Thread-safe getter for the value. If no value is present, one will be produced
        and this getter blocks until then.
        """
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
        """
        Returns an awaitable :py:class:`Future` that will hold the value once it is available.
        """
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
