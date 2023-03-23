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
    # result[:, 0] -= left
    # result[:, 1] -= top
    # result[:, 0] *= height_ratio
    # result[:, 1] *= width_ratio

    result[:, 0] -= left
    result[:, 1] -= top
    result[:, 0] *= width_ratio
    result[:, 1] *= height_ratio
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


# def get_torch_image_cut_box(leftTop, rightBottom, ExpandsRatio):
#     leftTop = np.maximum(leftTop, [0, 0])
#     rightBottom = np.minimum(rightBottom, [1920, 1080])
#
#
#     def _expand_crop_box(lt, rb, scale):
#         center = (lt + rb) / 2.0
#         xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]
#         xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
#         # expand it
#         lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
#         lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
#
#         lt = np.maximum(lt, [0, 0])
#         rb = np.minimum(rb, [1920, 1080])
#         lb[0] = np.maximum(lb[0], 0)
#         lb[1] = np.minimum(lb[1], 1080)
#         rt[0] = np.minimum(rt[0], 1920)
#         rt[1] = np.maximum(rt[1], 0)
#
#         center = (lt + rb) / 2
#         return center, lt, rt, rb, lb
#
#
#
#     Center, leftTop, rightTop, rightBottom, leftBottom = _expand_crop_box(leftTop, rightBottom, ExpandsRatio)
#
#     offset = (rightBottom - leftTop) // 2
#
#     cx = offset[0]
#     cy = offset[1]
#
#     r = max(cx, cy)
#
#     cx = r
#     cy = r
#
#     x = int(Center[0])
#     y = int(Center[1])
#
#     # return [x - cx, y - cy], [x + cx, y + cy]
#     """
#     top (int) – Vertical component of the top left corner of the crop box.
#
#     left (int) – Horizontal component of the top left corner of the crop box.
#
#     height (int) – Height of the crop box.
#
#     width (int) – Width of the crop box.
#     """
#     left = int(x - cx)
#     top = int(y - cy)
#     height = int(2 * cx)
#     width = int(2 * cy)
#
#     return top, left, height, width

def get_torch_image_cut_box(left_top, right_bottom, expands_ratio, img_size=(1920, 1080)):
    left_top = np.maximum(left_top, [0, 0])
    right_bottom = np.minimum(right_bottom, img_size)

    center = (left_top + right_bottom) / 2.0
    expanded_dims = np.array(
        [(right_bottom[0] - left_top[0]) * expands_ratio[0], (right_bottom[1] - left_top[1]) * expands_ratio[2],
         (right_bottom[0] - left_top[0]) * expands_ratio[1], (right_bottom[1] - left_top[1]) * expands_ratio[3]])
    left_top = center - expanded_dims[[0, 1]] / 2
    right_bottom = center + expanded_dims[[2, 3]] / 2

    left_top = np.maximum(left_top, [0, 0])
    right_bottom = np.minimum(right_bottom, img_size)

    width, height = right_bottom - left_top
    max_dim = max(width, height)

    left = int(center[0] - max_dim / 2)
    top = int(center[1] - max_dim / 2)
    width = int(max_dim)
    height = int(max_dim)

    return top, left, height, width


def get_torch_image_cut_box_leres(left_top, right_bottom, expands_ratio, img_size=(1080, 1920)):
    # Clip the coordinates to the image bounds
    left_top = np.clip(left_top, [0, 0], img_size)
    right_bottom = np.clip(right_bottom, [0, 0], img_size)
    left_top = np.minimum(left_top, right_bottom)

    # Calculate the center and dimensions of the box
    center = (left_top + right_bottom) / 2
    box_dims = right_bottom - left_top

    # Expand the box based on the expands_ratio
    expanded_dims = np.array([box_dims[0] * expands_ratio[0], box_dims[1] * expands_ratio[2],
                              box_dims[0] * expands_ratio[1], box_dims[1] * expands_ratio[3]])
    expanded_left_top = center - expanded_dims[[0, 1]] / 2
    expanded_right_bottom = center + expanded_dims[[2, 3]] / 2

    # Clip the expanded coordinates to the image bounds
    # expanded_left_top = np.maximum(expanded_left_top, [0, 0])
    # expanded_right_bottom = np.minimum(expanded_right_bottom, img_size)

    expanded_left_top = np.clip(expanded_left_top, [0, 0], img_size)
    expanded_right_bottom = np.clip(expanded_right_bottom, [0, 0], img_size)
    expanded_left_top = np.minimum(expanded_left_top, expanded_right_bottom)

    # Calculate the final dimensions
    height, width = expanded_right_bottom - expanded_left_top

    return int(expanded_left_top[1]), int(expanded_left_top[0]), int(height), int(width)


# def get_torch_image_cut_box_new(left_top, right_bottom, image_size=(1080,1920), aspect_ratio=1.0, area_ratio=4):
#     left_top = np.clip(left_top, [0, 0], image_size)
#     right_bottom = np.clip(right_bottom, [0, 0], image_size)
#
#     left, top = left_top
#     right, bottom = right_bottom
#     image_height, image_width = image_size
#
#     # 人物框的宽度、高度和面积
#     person_width = right - left
#     person_height = bottom - top
#     person_area = person_width * person_height
#
#     # 计算新框有效区域的面积和尺寸
#     new_box_area = person_area * area_ratio
#     new_box_width = np.sqrt(new_box_area / aspect_ratio)
#     new_box_height = new_box_width * aspect_ratio
#
#     # 限制新框有效区域尺寸不超过原图尺寸
#     new_box_width = min(image_width, new_box_width)
#     new_box_height = min(image_height, new_box_height)
#
#     # 确保人物框在新框有效区域内
#     left_shift_range = min(0, left - (new_box_width - person_width))
#     top_shift_range = min(0, top - (new_box_height - person_height))
#     right_shift_range = max(0, (new_box_width - person_width) - left)
#     bottom_shift_range = max(0, (new_box_height - person_height) - top)
#
#     # 随机生成新框的左上角坐标
#     left_shift = np.random.uniform(left_shift_range, right_shift_range)
#     top_shift = np.random.uniform(top_shift_range, bottom_shift_range)
#     new_left = left - left_shift
#     new_top = top - top_shift
#
#     return new_top, new_left, new_box_height, new_box_width

def get_torch_image_cut_box_new(left_top, right_bottom, image_size=(1920,1080), aspect_ratio=1.0, area_ratio=2.0):
    left_top = np.clip(left_top, [0, 0], image_size)
    right_bottom = np.clip(right_bottom, [0, 0], image_size)


    left, top = left_top
    right, bottom = right_bottom
    image_width,image_height = image_size

    # 人物框的宽度、高度和面积
    person_width = right - left
    person_height = bottom - top
    person_area = person_width * person_height

    # 计算新框有效区域的面积和尺寸
    new_box_area = person_area * area_ratio
    new_box_width = np.sqrt(new_box_area / aspect_ratio)
    new_box_height = new_box_width * aspect_ratio

    # 如果新框有效区域尺寸超过原图尺寸，优先满足长宽比
    if new_box_width > image_width or new_box_height > image_height:
        new_box_width = min(image_width, new_box_width)
        new_box_height = min(image_height, new_box_height)

        # 重新计算有效区域的长度和宽度，使其满足长宽比
        if new_box_width / new_box_height > aspect_ratio:
            new_box_width = new_box_height * aspect_ratio
        else:
            new_box_height = new_box_width / aspect_ratio

    left_range = max(0, right - new_box_width)
    top_range = max(0, bottom - new_box_height)
    right_range = min(image_width - new_box_width, left)
    # right_range = left
    bottom_range = min(image_height - new_box_height, top)

    # 随机生成新框的左上角坐标
    new_left = np.random.uniform(left_range, right_range)
    new_top = np.random.uniform(top_range, bottom_range)

    return int(new_top), int(new_left), int(new_box_height), int(new_box_width)
