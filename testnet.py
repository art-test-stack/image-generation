from model.data import StackedMNIST
from model.test.verification_net import VerificationNet

from model.settings import *

if __name__=="__main__":
    trainset = StackedMNIST(train=True, mode=AE_DATAMODE)
    testset = StackedMNIST(train=False, mode=AE_DATAMODE)

    verifNet = VerificationNet()

    verifNet.force_relearn = True

    verifNet.train(trainset=trainset, valset=testset)