import sys
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv


def setup_openAI_client(api_key, organization):
    return openai.OpenAI(api_key=api_key, organization=organization)


def add_message_to_the_conversation(list_of_messages: list, role, content):
    list_of_messages.append({"role": role, "content": content})


def get_user_input():
    return input("User: ")


def get_ai_output(client: OpenAI, list_of_messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=list_of_messages
    )
    if response.choices[0].finish_reason == "length":
        sys.exit("The conversation is too long, please start a new one.")
    return response


def print_ai_output(ai_output):
    print("AI:", ai_output)


def validate_environment_variables(api_key, organization):
    if not api_key or not organization:
        sys.exit("API key or organization not provided in environment variables.")


def validate_system_arguments():
    if len(sys.argv) != 2:
        sys.exit("Missing command line argument")


def setup_and_validate_environment():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    organization = os.getenv("ORGANIZATION")
    validate_environment_variables(api_key, organization)
    return api_key, organization


def main():

    validate_system_arguments()
    api_key, organization = setup_and_validate_environment()
    client = setup_openAI_client(api_key, organization)

    list_of_messages = [
        {
            "role": "system",
            "content": f"You are a friendly and helpful assistant. Your name is {sys.argv[1]}",
        }
    ]
    total_tokens = 0

    while True:
        try:
            user_input = get_user_input()
            add_message_to_the_conversation(list_of_messages, "user", user_input)

            ai_output = get_ai_output(client, list_of_messages)
            ai_message = ai_output.choices[0].message.content
            add_message_to_the_conversation(list_of_messages, "assistant", ai_message)

            print_ai_output(ai_message)

            total_tokens += ai_output.usage.total_tokens
        except openai.APIConnectionError as e:
            sys.exit(f"Network error occurred: {e}")
        except (EOFError, KeyboardInterrupt):
            sys.exit(f"\nBye!\nTotal amount of tokens used: {total_tokens}")


if __name__ == "__main__":
    main()
