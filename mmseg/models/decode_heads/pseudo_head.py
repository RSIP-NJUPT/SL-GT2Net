# Copyright (c) OpenMMLab. All rights reserved.
# import numpy as np
# import torch.nn as nn
# from mmcv.cnn import ConvModule
import torch
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from typing import List, Tuple


@HEADS.register_module()
class PseudoHead(BaseDecodeHead):
	def __init__(self, num_classes, in_index=-1, **kwargs):
		meaningless_value = 8
		super().__init__(in_channels=meaningless_value, 
                         channels=meaningless_value, 
                         num_classes=num_classes,
						 in_index=in_index, 
                         **kwargs)
	def forward(self, inputs):
		if isinstance(inputs, List):
			return inputs[self.in_index]
		elif isinstance(inputs, torch.Tensor):
			return inputs
