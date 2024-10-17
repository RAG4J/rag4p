import base64
import configparser
import os
from configparser import ConfigParser

import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad, pad


class KeyLoader:
    def __init__(self, key_path: str = None,
                 secret_key: str = None):
        self.key_path = key_path if key_path else "https://cocoen.nl/rag4jp/properties-encrypted.txt"
        self.secret_key = secret_key if secret_key else os.getenv("SECRET_KEY")

    def get_weaviate_api_key(self):
        return self.get_property('WEAVIATE_API_KEY')

    def get_weaviate_url(self):
        return self.get_property('WEAVIATE_URL')

    def get_openai_api_key(self):
        return self.get_property('OPENAI_API_KEY')

    def get_bedrock_region(self):
        return self.get_property('BEDROCK_REGION')

    def get_bedrock_profile(self):
        return self.get_property('BEDROCK_PROFILE')

    def get_property(self, key: str):
        value = os.environ.get(key)
        if not value:
            properties = self.load_properties_from_remote_file(self.key_path)
            value = properties.get(section="DEFAULT", option=key)

        if not value:
            raise Exception(f"Could not find property {key}")

        return value

    def decrypt(self, encrypted_text):
        try:
            encrypted_data = base64.b64decode(encrypted_text)
            cipher = AES.new(self.secret_key.encode(), AES.MODE_ECB)
            decrypted_data = cipher.decrypt(encrypted_data)
            unpadded_data = unpad(decrypted_data, AES.block_size)
            return unpadded_data.decode('utf-8')
        except Exception as e:
            print(f"There was a problem while decrypting the key: {str(e)}")
            raise

    def encrypt(self, str_to_encrypt):
        try:
            cipher = AES.new(self.secret_key.encode(), AES.MODE_ECB)
            padded_data = pad(str_to_encrypt.encode(), AES.block_size)
            encrypted_data = cipher.encrypt(padded_data)
            return base64.b64encode(encrypted_data).decode('utf-8')
        except Exception as e:
            print(f"There was a problem while encrypting the key: {str(e)}")
            raise

    def load_properties_from_remote_file(self, url) -> ConfigParser:
        """
        Load the properties from the remote file and decrypt the values, the file is written using java.util.Properties.
        It has a nasty habit of changing the values, so we need to replace the double backslashes with single ones.
        :param url: The url to load the properties from
        :return: The properties in the format of a ConfigParser
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise exception if the request failed

            lines = response.text.splitlines()
            lines = [line.replace('\\\\', '\\') for line in lines if not line.startswith('#')]
            properties_content = '\n'.join(lines)

            properties = ConfigParser()
            content = "[DEFAULT]\n" + properties_content
            properties.read_string(content)

            for line in lines:
                key, encrypted_value = line.split('=', 1)
                decrypted_value = self.decrypt(encrypted_value.strip())
                properties.set('DEFAULT', key.strip(), decrypted_value)

            return properties

        except (requests.RequestException, configparser.Error) as e:
            print(f"There was a problem while loading the properties file: {str(e)}")
            raise
