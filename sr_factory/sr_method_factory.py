__author__ = 'Sherwin'

from sr_method.iccv09 import ICCV09
from sr_exception.sr_exception import SRException

class SRMethodFactory(object):

    @classmethod
    def create_method(cls, method_type):
        """Create a SR method object.

        @param type: type of SR method
        @type method_type: str
        @return: an instance of SR method
        @rtype:
        """
        if (method_type == "iccv09"):
            return ICCV09()
        else:
            raise SRException("Unsupported SR Method Type:%s" % method_type)

