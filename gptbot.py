import sys
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv


def setup_openAI_client():
    load_dotenv()

    return OpenAI(
        api_key=os.getenv("API_KEY"),
        organization=os.getenv("ORGANIZATION"),
    )


def add_message_to_the_conversation(list_of_messages: list, role, content):
    list_of_messages.append({"role": role, "content": content})


def main():

    if len(sys.argv) != 2:
        sys.exit("Missing command line argument")

    client = setup_openAI_client()

    list_of_messages = [
        {
            "role": "system",
            "content": f"You are a friendly and helpful assistant. Your name is {sys.argv[1]}",
        }
    ]
    total_tokens = 0

    while True:
        try:
            user_input = input("User: ")
            add_message_to_the_conversation(list_of_messages, "user", user_input)

            response = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=list_of_messages
            )
            if response.choices[0].finish_reason == "length":
                sys.exit("The conversation is too long, please start a new one.")

            ai_output = response.choices[0].message.content
            total_tokens += response.usage.total_tokens

            add_message_to_the_conversation(list_of_messages, "assistant", ai_output)

            print("AI:", ai_output)
            continue
        except requests.RequestException as e:
            sys.exit(f"Network error occurred: {e}")
        except EOFError:
            sys.exit(f"\nBye!\nTotal amount of tokens used: {total_tokens}")


if __name__ == "__main__":
    main()
