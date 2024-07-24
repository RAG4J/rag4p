class QuestionAnswerRecord:
    def __init__(self, document_id: str, chunk_id: str, chunk_text: str, question: str):
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.chunk_text = chunk_text
        self.question = question
