import os

from rag4p.generation.observed_answer_generator import ObservedAnswerGenerator
from rag4p.connectopenai.openai_answer_generator import OpenaiAnswerGenerator
from rag4p.tracker.rag_tracker import global_data
from rag4p.util.key_loader import KeyLoader


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()

    question = "Since when was the Vasa available for the public to visit?"
    context = ("By Friday 16 February 1962, the ship is ready to be displayed to the general public at the "
               "newly-constructed Wasa Shipyard, where visitors can see Vasa while a team of conservators, "
               "carpenters and other technicians work to preserve the ship.")

    openai_answer_generator = OpenaiAnswerGenerator(openai_api_key=key_loader.get_openai_api_key())
    answer_generator = ObservedAnswerGenerator(answer_generator=openai_answer_generator)
    answer = answer_generator.generate_answer(question, context)

    print(f"Question: {global_data['observer'].question}")
    print(f"Context: {global_data['observer'].context}")
    print(f"Answer: {answer}")

