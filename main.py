from chatbot import initialize_bot
from chatbot import IOManager

import time


def start_bot():
    initialize_bot()


def generate_reply(user_input):
    manager = IOManager()
    manager.set_input(user_input)
    while True:
        if manager.is_response_generated():
            break
        else:
            time.sleep(0.05)
    response = manager.get_response()
    print(response)


start_bot()
generate_reply("Hello")
