class Interface:
    @staticmethod
    def init():
        pass

    @staticmethod
    def release():
        pass

    @staticmethod
    def update_stateless(pose, actions, **kwargs):
        raise NotImplementedError
