class DS_Template():
    def __init__(self):
        self.TRAIN_SET = 0
        self.VALID_SET = 1
        self.TEST_SET = 2
        self.TRAIN_MODE = 0
        self.VALID_MODE = 1
        self.TEST_MODE = 2

        self.MIN_QUEUE_SIZE = 10 #TF-Record minimal Queu-size
        self.MAX_QUEUE_SIZE = 20 #TF-Record maximal Queu-size
