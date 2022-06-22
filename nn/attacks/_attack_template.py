class Attack_Template():
    def __init__(self):
        self.RANDOM = 0
        self.FGM = 1
        self.FGSM = 2
        self.LCM = 3
        self.PGD = 4
        self.PGD_STEP = 5
        self.DEEPFOOL = 6
        self.CW = 7
        self.JSMA = 8
        self.LOCALSEARCH = 9
        self.SPSA = 10
        self.SIMBA = 11
        self.GENATTACK = 12
        self.ZOO = 13

    def convert_name(self, name):
        if name == "RANDOM":
            return self.RANDOM
        if name == "FGM":
            return self.FGM
        if name == "FGSM":
            return self.FGSM
        if name == "LCM":
            return self.LCM
        if name == "PGD":
            return self.PGD
        if name == "PGD_STEP":
            return self.PGD_STEP
        if name == "DEEPFOOL":
            return self.DEEPFOOL
        if name == "CW":
            return self.CW
        if name == "JSMA":
            return self.JSMA
        if name == 'SPSA':
            return self.SPSA
        if name == 'SIMBA':
            return self.SIMBA
        if name == "LOCALSEARCH":
            return self.LOCALSEARCH
        if name == "GENATTACK":
            return self.GENATTACK
        if name == "ZOO":
            return self.ZOO
