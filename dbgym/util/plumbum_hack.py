from plumbum.machines import LocalCommand, RemoteCommand

class PlumbumQuoteHack:
    """Patches around crazy quoting bugs in plumbum; manually quote instead."""
    def __init__(self):
        self._local_quote_level = None
        self._remote_quote_level = None

    def __enter__(self):
        self._local_quote_level = LocalCommand.QUOTE_LEVEL
        self._remote_quote_level = RemoteCommand.QUOTE_LEVEL
        LocalCommand.QUOTE_LEVEL = 999
        RemoteCommand.QUOTE_LEVEL = 999

    def __exit__(self, exc_type, exc_val, exc_tb):
        LocalCommand.QUOTE_LEVEL = self._local_quote_level
        RemoteCommand.QUOTE_LEVEL = self._remote_quote_level