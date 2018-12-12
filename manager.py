class IOManager(object):

    user_input = None
    response = None

    @staticmethod
    def is_input_set():
        if IOManager.user_input is not None:
            return True
        else:
            return False

    @staticmethod
    def get_input():
        return IOManager.user_input

    @staticmethod
    def set_input(user_in):
        IOManager.user_input = user_in
        IOManager.response = None

    @staticmethod
    def add_response(generated_reply):
        IOManager.response = generated_reply
        IOManager.user_input = None

    @staticmethod
    def is_response_generated():
        if IOManager.response is not None:
            return True
        else:
            return False

    @staticmethod
    def get_response():
        return IOManager.response
