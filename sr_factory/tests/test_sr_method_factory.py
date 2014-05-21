from sr_factory.sr_method_factory import SRMethodFactory

__author__ = 'Sherwin'

import unittest
from sr_exception.sr_exception import SRException

class TestSRMethodFactory(unittest.TestCase):

    def test_create_sr_method(self):
        sr_method = SRMethodFactory.create_method("iccv09")
        self.assertEqual("iccv09", sr_method.get_method_type())
        with self.assertRaises(SRException):
            SRMethodFactory.create_method("invalid method type")

if __name__ == "__main__":
    unittest.main()