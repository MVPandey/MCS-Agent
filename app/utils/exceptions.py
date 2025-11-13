class ChatModelError(Exception):
    """Exception for chat model errors."""

    def __init__(self, message: str):
        """Initialize the chat model error."""
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return the string representation of the chat model error."""
        return self.message
