import pytest
from unittest.mock import patch
from gptbot import add_message_to_the_conversation, main, setup_openAI_client


def test_add_message_to_the_conversation():
    messages = []
    add_message_to_the_conversation(messages, "user", "Hello, world!")
    assert messages == [{"role": "user", "content": "Hello, world!"}]


def test_main_missing_argument():
    with patch("sys.argv", ["gptbot.py"]), pytest.raises(SystemExit) as e:
        main()
    assert str(e.value) == "Missing command line argument"
