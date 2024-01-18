import unittest
from unittest.mock import patch, MagicMock
from rag4p.util.key_loader import KeyLoader


class TestKeyLoader(unittest.TestCase):

    @patch('os.environ', {'API_KEY': '12345'})
    def test_get_property_from_environment(self):
        loader = KeyLoader()
        self.assertEqual(loader.get_property('API_KEY'), '12345')

    @patch('os.environ', {})
    @patch('rag4p.util.key_loader.KeyLoader.load_properties_from_remote_file')
    def test_get_property_from_remote_file_mock(self, mock_load):
        mock_load.return_value.get.return_value = '67890'
        loader = KeyLoader(key_path='dummy_path')
        self.assertEqual(loader.get_property('API_KEY'), '67890')

    @patch('os.environ', {})
    def test_get_property_from_remote_file(self):
        loader = KeyLoader(key_path='https://www.retrocinevr.nl/rag4j/properties-encrypted-test.txt',
                           secret_key='thisisjustatestkeythatweneednow9')
        self.assertEqual('test-key-openai', loader.get_openai_api_key())

    def test_encrypt_property(self):
        loader = KeyLoader(key_path='https://www.retrocinevr.nl/rag4j/properties-encrypted-test.txt',
                           secret_key='thisisjustatestkeythatweneednow9')
        encrypted = loader.encrypt('https://weaviate.org')
        self.assertEqual('wtKBw2n9PR2dV55hsZIh8dUWRtCCBn+4bHdbtGysZIM=', encrypted)

    def test_decrypt_property(self):
        loader = KeyLoader(key_path='https://www.retrocinevr.nl/rag4j/properties-encrypted-test.txt',
                           secret_key='thisisjustatestkeythatweneednow9')
        encrypted = loader.decrypt(R'4pfWfe2YQ4VApdq7gntMiw==')
        self.assertEqual('test-key-openai', encrypted)

    @patch('os.environ', {})
    @patch('rag4p.util.key_loader.KeyLoader.load_properties_from_remote_file')
    def test_get_property_not_found(self, mock_load):
        mock_load.return_value.get.return_value = None
        loader = KeyLoader(key_path='dummy_path')
        with self.assertRaises(Exception):
            loader.get_property('API_KEY')

    @patch('rag4p.util.key_loader.requests.get')
    def test_load_properties_from_remote_file(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = '# just another comment\nAPI_KEY=4pfWfe2YQ4VApdq7gntMiw\\\\=\\\\='
        mock_get.return_value = mock_response
        loader = KeyLoader('dummy_path', secret_key='thisisjustatestkeythatweneednow9')
        properties = loader.load_properties_from_remote_file('dummy_url')
        self.assertEqual(properties.get('DEFAULT', 'API_KEY'), 'test-key-openai')

    @patch('rag4p.util.key_loader.requests.get')
    def test_load_properties_from_remote_file_failure(self, mock_get):
        mock_get.side_effect = Exception('Failed to load')
        with self.assertRaises(Exception):
            KeyLoader.load_properties_from_remote_file('dummy_url')


if __name__ == '__main__':
    unittest.main()
