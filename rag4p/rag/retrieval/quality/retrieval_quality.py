class RetrievalQuality:
    def __init__(self, correct: [str], incorrect: [str]):
        self.correct = correct
        self.incorrect = incorrect

    def precision(self):
        return len(self.correct) / (len(self.correct) + len(self.incorrect))

    def total_questions(self):
        return len(self.correct) + len(self.incorrect)