
def mask(annotations, data_shape):
    import skimage.draw, numpy as np
    from skimage.morphology import square
    # Use a breakpoint in the code line below to debug your script.
    lines = annotations[0][3]
    points = annotations[0][1]

    webmask = np.full(data_shape, False, dtype=np.bool_)
    for line in lines:
        rr, cc, val = skimage.draw.line_aa(line[0], line[1], line[2], line[3])
        # idx1 = np.argwhere(rr>=1024)
        # idx2 = np.argwhere(cc>=1024)
        # if np.size(idx2)==0:
        #    idx = idx1
        # else:
        #    idx = idx1 or idx2
        # cc  = np.delete(cc, idx)
        # rr  = np.delete(rr, idx)
        webmask[rr, cc] = True

    for point in points:
        if point[0] >= 1024 or point[1] >= 1024:
            continue
        webmask[point[0], point[1]] = True

    webmask = skimage.morphology.dilation(webmask, square(3)) # Press Ctrl+F8 to toggle the breakpoint.

    return webmask

