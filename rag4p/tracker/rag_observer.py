class RAGObserver:
    question: str
    answer: str
    context: str

    def __init__(self, question: str = "", answer: str = "", context: str = ""):
        self.question = question
        self.answer = answer
        self.context = context
        self.relevant_chunks = {}
        self.relevant_chunks_with_windows = {}
        self.relevant_chunks_windows_text = {}

    def add_relevant_chunk(self, chunk_id: str, text: str):
        """Add a relevant chunk to the observer. Chunk id is the document id + "_" + chunk id."""
        self.relevant_chunks[chunk_id] = text

    def add_window_to_relevant_chunk(self, chunk_id: str, window: [int]):
        """Add a window to a relevant chunk. Chunk id is the document id + "_" + chunk id."""
        self.relevant_chunks[chunk_id] = window

    def add_window_text_to_relevant_chunk(self, chunk_id: str, window_text: str):
        """Add a window text to a relevant chunk. Chunk id is the document id + "_" + chunk id."""
        self.relevant_chunks_windows_text[chunk_id] = window_text

    def reset(self):
        """Reset the observer."""
        self.question = ""
        self.answer = ""
        self.context = ""
        self.relevant_chunks = {}
        self.relevant_chunks_with_windows = {}
        self.relevant_chunks_windows_text = {}