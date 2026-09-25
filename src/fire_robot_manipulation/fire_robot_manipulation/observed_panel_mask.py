"""Conservative color support for the currently observed colored door panel."""
import numpy as np


def panel_color_support(bgr, color):
    image=np.asarray(bgr)
    if image.ndim!=3 or image.shape[2]!=3 or color not in ('blue','green','red'):
        raise ValueError('unsupported_panel_color_or_image')
    channels=image.astype(float)
    index={'blue':0,'green':1,'red':2}[color]
    primary=channels[:,:,index]
    others=np.max(channels[:,:,[i for i in range(3) if i!=index]],axis=2)
    mask=(primary>45)&(primary>1.4*others)&(primary-others>25)
    # Remove color/depth silhouette mixing at the panel boundary.
    for _ in range(2):
        padded=np.pad(mask,1,constant_values=False)
        mask=np.logical_and.reduce([padded[y:y+image.shape[0],x:x+image.shape[1]]
                                    for y in range(3) for x in range(3)])
    return mask
