from nn.attacks._attack_template import Attack_Template


def fetch_attack(session_config):

    AA_content = Attack_Template()
    method_name = session_config['Attack_config']['method']
    method = AA_content.convert_name(method_name)

    if   method == AA_content.FGM:          from nn.attacks.white_box.FGM import FGM as Attack
    elif method == AA_content.FGSM:         from nn.attacks.white_box.FGSM import FGSM as Attack
    elif method == AA_content.LCM:          from nn.attacks.white_box.LCM import LCM as Attack
    elif method == AA_content.PGD:          from nn.attacks.white_box.PGD import PGD as Attack
    elif method == AA_content.DEEPFOOL:     from nn.attacks.white_box.Deepfool import Deepfool as Attack
    elif method == AA_content.CW:           from nn.attacks.white_box.CW import CW as Attack
    elif method == AA_content.JSMA:         from nn.attacks.white_box.JSMA import JSMA as Attack
    elif method == AA_content.SPSA:         from nn.attacks.black_box.SPSA import SPSA as Attack
    elif method == AA_content.SIMBA:        from nn.attacks.black_box.SimBA import SimBA as Attack
    elif method == AA_content.LOCALSEARCH:  from nn.attacks.black_box.LocalSearch import LocalSearch as Attack
    elif method == AA_content.GENATTACK:    from nn.attacks.black_box.GenAttack import GenAttack as Attack
    elif method == AA_content.ZOO:          from nn.attacks.black_box.ZOO import ZOO as Attack
    else:                                   from nn.attacks.black_box.Random import Random as Attack

    return Attack