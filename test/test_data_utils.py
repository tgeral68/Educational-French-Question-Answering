import unittest
import torch

from src.data_utils import misc 

class TestDataUtils(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_padding_float(self):
        x = torch.rand(5, 4)
        padded_x = misc.Padding(padding_value=27, dim=-1)(x, 10)
        assert(padded_x.shape[0] == 5 and padded_x.shape[1] == 10)
        assert(torch.all(padded_x[:, 4:] == 27))
        assert(torch.all(padded_x[:, :4] == x))
    
    def test_padding_bool(self):
        x = torch.BoolTensor(5, 4)
        padded_x = misc.Padding(padding_value=True, dim=-1)(x, 10)
        assert(padded_x.shape[0] == 5 and padded_x.shape[1] == 10)
        assert(torch.all(padded_x[:, 4:] == True))
        assert(torch.all(padded_x[:, :4] == x))

    def test_padding_byte(self):
        x = torch.ByteTensor(5, 4)
        padded_x = misc.Padding(padding_value=27, dim=-1)(x, 10)
        assert(padded_x.shape[0] == 5 and padded_x.shape[1] == 10)
        assert(torch.all(padded_x[:, 4:] == 27))
        assert(torch.all(padded_x[:, :4] == x))

    def test_padding_int(self):
        x = torch.LongTensor(5, 4)
        padded_x = misc.Padding(padding_value=27, dim=-1)(x, 10)
        assert(padded_x.shape[0] == 5 and padded_x.shape[1] == 10)
        assert(torch.all(padded_x[:, 4:] == 27))
        assert(torch.all(padded_x[:, :4] == x))
    
    def test_collatePadding_float(self):
        x = [torch.rand(5, 4), torch.rand(7, 8), torch.rand(3, 2)]
        collate_padded_x = misc.CollatePadding()(x)
        assert(collate_padded_x.shape[0] == 15 and  collate_padded_x.shape[1] == 8)

    def test_collatePadding_bool(self):
        x = [torch.BoolTensor(5, 4), torch.BoolTensor(7, 8), torch.BoolTensor(3, 2)]
        collate_padded_x = misc.CollatePadding(1)(x)
        assert(collate_padded_x.shape[0] == 15 and  collate_padded_x.shape[1] == 8)

