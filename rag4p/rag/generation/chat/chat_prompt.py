
class ChatPrompt:
    def __init__(self,
                 system_message: str = None,
                 user_message: str = None,
                 system_message_filename: str = None,
                 user_message_filename: str = None):
        self.system_message = system_message
        self.user_message = user_message
        if system_message_filename:
            self.system_message = self.__read_from_file(system_message_filename)
        if user_message_filename:
            self.user_message = self.__read_from_file(user_message_filename)

    def create_system_message(self, params: dict):
        return self.system_message.format(**params)

    def create_user_message(self, params: dict):
        return self.user_message.format(**params)

    def __read_from_file(self, filename: str):
        with open(filename, 'r') as file:
            return file.read()