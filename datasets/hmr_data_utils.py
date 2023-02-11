import numpy as np


def scale_kpts(kpts, height_ratio, width_ratio):
    # ratio = 1.0 * args.crop_size / image.shape[0]
    result = kpts.copy()
    result[:, 0] *= height_ratio
    result[:, 1] *= width_ratio
    return result


def off_set_kpts(kpts, leftTop):
    result = kpts.copy()
    result[:, 0] -= leftTop[0]
    result[:, 1] -= leftTop[1]
    return result


def off_set_scale_kpts(kpts, top, left, height_ratio, width_ratio):
    result = kpts.copy()
    result[:, 0] -= left
    result[:, 1] -= top
    result[:, 0] *= height_ratio
    result[:, 1] *= width_ratio
    return result


def calc_aabb(kpts):
    ptLeftTop = np.array([kpts[0][0], kpts[0][1]])
    ptRightBottom = ptLeftTop.copy()
    for kpt_i in kpts:
        ptLeftTop[0] = min(ptLeftTop[0], kpt_i[0])
        ptLeftTop[1] = min(ptLeftTop[1], kpt_i[1])
        ptRightBottom[0] = max(ptRightBottom[0], kpt_i[0])
        ptRightBottom[1] = max(ptRightBottom[1], kpt_i[1])

    return ptLeftTop, ptRightBottom


def collect_valid_kpts(kpts):
    valid_kpts = []
    for kpt in kpts:
        if kpt[2] != 0:
            valid_kpts.append(kpt)
    return valid_kpts



def get_torch_image_cut_box(leftTop, rightBottom, ExpandsRatio, Center=None):
    leftTop = np.maximum(leftTop, [0, 0])
    rightBottom = np.minimum(rightBottom, [1920,1080])


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


        lt = np.maximum(lt, [0, 0])
        rb = np.minimum(rb, [1920, 1080])
        lb[0]=np.maximum(lb[0],0)
        lb[1]=np.minimum(lb[1],1080)
        rt[0]=np.minimum(rt[0],1920)
        rt[1]=np.maximum(rt[1],0)


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
    left = int(x - cx)
    top = int(y - cy)
    height = int(2 * cx)
    width = int(2 * cy)
    return top, left, height, width