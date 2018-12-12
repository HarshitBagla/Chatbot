from flask import Flask
import main as m

chatbotapi = Flask(__name__)


@chatbotapi.route('/')
def index():
    m.start_bot()


@chatbotapi.route('/reply/<string:text>/', methods=['GET', 'POST'])
def reply(text):
    try:
        response = m.generate_reply(text)
        print(response)
        return response
    except Exception:
        print("Problem")


if __name__ == '__main__':
    chatbotapi.run(debug=True)
