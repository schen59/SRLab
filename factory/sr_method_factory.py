__author__ = 'Sherwin'

from iccv09 import ICCV09
from sr_exception.sr_exception import SRException

class SRMethodFactory(object):

    @classmethod
    def createMethod(cls, type):
        if (type == "iccv09"):
            return ICCV09()
        else:
            raise SRException("Unsupported SR Method Type:%s" % type)

