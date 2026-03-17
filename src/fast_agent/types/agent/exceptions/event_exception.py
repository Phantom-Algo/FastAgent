class EventChannelClosedException(Exception):
    """Raised when trying to send an event to a closed channel."""
    pass