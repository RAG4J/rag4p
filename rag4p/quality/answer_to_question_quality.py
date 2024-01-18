class AnswerToQuestionQuality:
    """
    Represents the quality of an answer to a question. The quality is a number between 1 and 5, where 5
    means that the answer is a complete answer to the question and 1 means that the answer unrelated to the
    question.
    """
    quality: int
    reason: str

    def __init__(self, quality: int, reason: str):
        self.quality = quality
        self.reason = reason
