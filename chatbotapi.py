from flask import Flask
import main as m

chatbotapi = Flask(__name__)


@chatbotapi.route('/')
def index():
    m.start_bot()


@chatbotapi.route('/reply', methods=['GET', 'POST'])
def reply(user_input):
    return m.generate_reply(user_input)


if __name__ == '__main__':
    chatbotapi.run(debug=True)
