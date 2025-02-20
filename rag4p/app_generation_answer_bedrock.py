from rag4p.integrations.bedrock.access_bedrock import AccessBedrock
from rag4p.integrations.bedrock.bedrock_answer_generator import BedrockAnswerGenerator
from rag4p.logging_config import setup_logging
from rag4p.rag.generation.observed_answer_generator import ObservedAnswerGenerator
from rag4p.rag.tracker.rag_tracker import global_data
from rag4p.util.key_loader import KeyLoader


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    key_loader = KeyLoader()

    question = "Since when was the Vasa available for the public to visit?"
    context = ("By Friday 16 February 1962, the ship is ready to be displayed to the general public at the "
               "newly-constructed Wasa Shipyard, where visitors can see Vasa while a team of conservators, "
               "carpenters and other technicians work to preserve the ship.")

    access_bedrock = AccessBedrock()
    bedrock_answer_generator = BedrockAnswerGenerator(access_bedrock=access_bedrock)
    answer_generator = ObservedAnswerGenerator(answer_generator=bedrock_answer_generator)
    answer = answer_generator.generate_answer(question, context)

    print(f"Question: {global_data['observer'].question}")
    print(f"Context: {global_data['observer'].context}")
    print(f"Answer: {answer}")

