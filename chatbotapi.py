from flask import Flask
import main as m

chatbotapi = Flask(__name__)

@chatbotapi.route('/')

def index():
    m.start_bot()

@chatbotapi.route('/reply', methods=['GET', 'POST'])

def reply(str):
    return m.generate_reply(str)



if __name__ == '__main__':
	app.run(debug=True)
