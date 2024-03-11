import os
import unittest
from unittest.mock import patch, mock_open
from rag4p.rag.generation.chat.chat_prompt import ChatPrompt


class TestChatPrompt(unittest.TestCase):
    def setUp(self):
        self.chat_prompt = ChatPrompt(system_message="Hello, {name}!", user_message="Hi, {name}!")

    def test_create_system_message_with_valid_params(self):
        result = self.chat_prompt.create_system_message({"name": "Alice"})
        self.assertEqual(result, "Hello, Alice!")

    def test_create_system_message_with_missing_params(self):
        with self.assertRaises(KeyError):
            self.chat_prompt.create_system_message({})

    def test_create_user_message_with_valid_params(self):
        result = self.chat_prompt.create_user_message({"name": "Bob"})
        self.assertEqual(result, "Hi, Bob!")

    def test_create_user_message_with_missing_params(self):
        with self.assertRaises(KeyError):
            self.chat_prompt.create_user_message({})

    @patch("builtins.open", new_callable=mock_open, read_data="Hello, {name}!")
    def test_read_system_message_from_file(self, mock_file):
        chat_prompt = ChatPrompt(system_message_filename="system.txt")
        result = chat_prompt.create_system_message({"name": "Alice"})
        self.assertEqual(result, "Hello, Alice!")

    def test_read_user_message_from_file(self):
        directory = os.getcwd()
        file_path = os.path.join(directory, "data/quality", "quality_of_answer_from_context_user.txt")

        chat_prompt = ChatPrompt(user_message_filename=file_path)
        result = chat_prompt.create_user_message({"answer": "my answer", "context": "provided context"})
        self.assertEqual("Answer: my answer\\nContext: provided context\\nResult:\\n", result)


if __name__ == '__main__':
    unittest.main()