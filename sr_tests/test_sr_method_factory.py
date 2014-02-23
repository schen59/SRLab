from factory.sr_method_factory import SRMethodFactory

__author__ = 'Sherwin'

import unittest
from sr_exception.sr_exception import SRException

class TestSRMethodFactory(unittest.TestCase):

    def testCreateMethod(self):
        sr_method = SRMethodFactory.createMethod("iccv09")
        self.assertEqual("iccv09", sr_method.getType())
        with self.assertRaises(SRException):
            SRMethodFactory.createMethod("invalid method type")

if __name__ == "__main__":
    unittest.main()