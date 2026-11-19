import unittest

from backbone_test_utils import BackboneWorkflowMixin


class StackedHourglassWorkflowTests(BackboneWorkflowMixin, unittest.TestCase):
    backbone_name = "StackedHourglass"
