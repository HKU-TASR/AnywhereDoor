from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch

from mmengine.infer.infer import ModelType
from mmdet.apis import DetInferencer
from mmdet.AnywhereDoor.modify_image_funcs import get_modified_image_repeat

class BackdoorInferencer(DetInferencer):
    def __init__(self,
                 model: Optional[Union[ModelType, str]] = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: Optional[str] = 'mmdet',
                 palette: str = 'none',
                 show_progress: bool = True) -> None:
        super().__init__(model, weights, device, scope, palette, show_progress)
        self.mask = None

    @torch.no_grad()
    def forward(self, inputs: Union[dict, tuple], **kwargs) -> Any:
        ###############################################################################################
        ###     modify image
        if self.mask is None:
            self.mask = torch.zeros(3, 30, 30).to('cuda:0')
        inputs['inputs'][0] = get_modified_image_repeat(inputs['inputs'][0].to('cuda:0'), self.mask)

        ###############################################################################################
        return super().forward(inputs, **kwargs)