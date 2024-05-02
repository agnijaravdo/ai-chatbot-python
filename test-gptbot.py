import pytest
from unittest.mock import patch

import requests
from gptbot import add_message_to_the_conversation, main, setup_openAI_client


def test_add_message_to_the_conversation():
    messages = []
    add_message_to_the_conversation(messages, "user", "Hello, world!")
    assert messages == [{"role": "user", "content": "Hello, world!"}]


def test_main_missing_argument():
    with patch("sys.argv", ["gptbot.py"]), pytest.raises(SystemExit) as e:
        main()
    assert str(e.value) == "Missing command line argument"


def test_setup_openAI_client():
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = ["api_key", "organization"]
        client = setup_openAI_client()
        assert client.api_key == "api_key"
        assert client.organization == "organization"


def test_main_network_error():
    with patch("sys.argv", ["gptbot.py", "Tom"]), patch(
        "gptbot.setup_openAI_client"
    ) as mock_setup_openAI_client, patch("gptbot.input") as mock_input:
        mock_input.side_effect = [requests.RequestException]
        with pytest.raises(SystemExit) as e:
            main()
        assert str(e.value) == f"Network error occurred: "


def test_main_eof_error():
    with patch("sys.argv", ["gptbot.py", "Tom"]), patch(
        "gptbot.setup_openAI_client"
    ) as mock_setup_openAI_client, patch("gptbot.input") as mock_input:
        mock_input.side_effect = [EOFError]
        with pytest.raises(SystemExit) as e:
            main()
        assert str(e.value) == f"\nBye!\nTotal amount of tokens used: 0"
