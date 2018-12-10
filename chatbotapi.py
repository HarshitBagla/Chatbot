from flask import Flask
import main as m

chatbotapi = Flask(__name__)


@chatbotapi.route('/')
def index():
    m.start_bot()

<<<<<<< HEAD
@chatbotapi.route('/reply/<string:text>/', methods=['GET', 'POST'])

def reply(text):
    return m.generate_reply(text)
=======
>>>>>>> def0e326beb909005854a75ea1dac1ceb6bac5fd

@chatbotapi.route('/reply', methods=['GET', 'POST'])
def reply(user_input):
    return m.generate_reply(user_input)


if __name__ == '__main__':
<<<<<<< HEAD
	chatbotapi.run(debug=True)
=======
    chatbotapi.run(debug=True)
>>>>>>> def0e326beb909005854a75ea1dac1ceb6bac5fd
