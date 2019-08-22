"""
Helper module used to tun tests
"""

import unittest
import xmlrunner

if __name__ == '__main__':
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test_reports'), module=None)
