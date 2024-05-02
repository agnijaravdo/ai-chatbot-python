import openai
import pytest
from unittest.mock import patch
from gptbot import (
    add_message_to_the_conversation,
    main,
    setup_openAI_client,
    get_user_input,
    print_ai_output,
    validate_environment_variables,
    validate_system_arguments,
    setup_and_validate_environment,
)


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
        client = setup_openAI_client("api_key", "organization")
        assert client.api_key == "api_key"
        assert client.organization == "organization"


def test_main_eof_error():
    with patch("sys.argv", ["gptbot.py", "Tom"]), patch(
        "gptbot.setup_openAI_client"
    ) as mock_setup_openAI_client, patch("gptbot.input") as mock_input:
        mock_input.side_effect = [EOFError]
        with pytest.raises(SystemExit) as e:
            main()
        assert str(e.value) == f"\nBye!\nTotal amount of tokens used: 0"


def test_main_keyboard_interrupt():
    with patch("sys.argv", ["gptbot.py", "Tom"]), patch(
        "gptbot.setup_openAI_client"
    ) as mock_setup_openAI_client, patch("gptbot.input") as mock_input:
        mock_input.side_effect = [KeyboardInterrupt]
        with pytest.raises(SystemExit) as e:
            main()
        assert str(e.value) == f"\nBye!\nTotal amount of tokens used: 0"


def test_main_api_connection_error():
    with patch("sys.argv", ["gptbot.py", "Tom"]), patch(
        "gptbot.setup_openAI_client"
    ) as mock_setup_openAI_client, patch("gptbot.input") as mock_input:
        mock_input.side_effect = [openai.APIConnectionError(request="request")]
        with pytest.raises(SystemExit) as e:
            main()
        assert (
            str(e.value)
            == f"Network error occurred: {openai.APIConnectionError(request='request')}"
        )


def test_get_user_input():
    with patch("builtins.input") as mock_input:
        mock_input.return_value = "Hello, world!"
        user_input = get_user_input()
        assert user_input == "Hello, world!"


def test_print_ai_output(capsys):
    ai_output = "Hello, world!"
    print_ai_output(ai_output)
    captured = capsys.readouterr()
    assert captured.out == f"AI: {ai_output}\n"


def test_validate_environment_variables():
    with pytest.raises(SystemExit) as e:
        validate_environment_variables("", "")
    assert (
        str(e.value) == "API key or organization not provided in environment variables."
    )


def test_setup_and_validate_environment():
    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = ["api_key", "organization"]
        api_key, organization = setup_and_validate_environment()
        assert api_key == "api_key"
        assert organization == "organization"


def test_validate_system_arguments_are_correct():
    with patch("sys.argv", ["gptbot.py", "Tom"]):
        validate_system_arguments()


def test_validate_system_arguments_are_missing():
    with patch("sys.argv", ["gptbot.py"]):
        with pytest.raises(SystemExit) as e:
            validate_system_arguments()
        assert str(e.value) == "Missing command line argument"
