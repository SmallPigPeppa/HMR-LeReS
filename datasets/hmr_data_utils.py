import numpy as np


def calc_aabb(ptSets):
    if not ptSets or len(ptSets) == 0:
        return False, False, False

    ptLeftTop = np.array([ptSets[0][0], ptSets[0][1]])
    ptRightBottom = ptLeftTop.copy()
    for pt in ptSets:
        ptLeftTop[0] = min(ptLeftTop[0], pt[0])
        ptLeftTop[1] = min(ptLeftTop[1], pt[1])
        ptRightBottom[0] = max(ptRightBottom[0], pt[0])
        ptRightBottom[1] = max(ptRightBottom[1], pt[1])

    return ptLeftTop, ptRightBottom, len(ptSets) >= 5

def collect_valid_kpts( kpts):
    valid_kpts = []
    for kpt in kpts:
        if kpt[2] != 0:
            valid_kpts.append(kpt)
    return valid_kpts

def get_torch_image_cut_box(leftTop, rightBottom, ExpandsRatio, Center=None):
    try:
        l = len(ExpandsRatio)
    except:
        ExpandsRatio = [ExpandsRatio, ExpandsRatio, ExpandsRatio, ExpandsRatio]

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]
        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        # expand it
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb

    if Center == None:
        Center = (leftTop + rightBottom) // 2

    Center, leftTop, rightTop, rightBottom, leftBottom = _expand_crop_box(leftTop, rightBottom, ExpandsRatio)

    offset = (rightBottom - leftTop) // 2

    cx = offset[0]
    cy = offset[1]

    r = max(cx, cy)

    cx = r
    cy = r

    x = int(Center[0])
    y = int(Center[1])

    # return [x - cx, y - cy], [x + cx, y + cy]
    """
    top (int) – Vertical component of the top left corner of the crop box.

    left (int) – Horizontal component of the top left corner of the crop box.
    
    height (int) – Height of the crop box.
    
    width (int) – Width of the crop box.
    """
    top = int(x - cx)
    left = int(y - cy)
    height = int(2 * cx)
    weight = int(2 * cy)
    return top, left, height, weight
