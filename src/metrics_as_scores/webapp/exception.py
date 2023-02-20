class PlotException(Exception):
    """
    This exception is used only in the web application in order to differentiate
    ordinary exceptions from those thrown "by design" in the web application (e.g.,
    when selecting a discrete random variable for a continuous quantity).
    """
    def __init__(self, msg: str, *args: object) -> None:
        self.msg = msg
        super().__init__(*args)
