class BBox(object):

    def __init__(self, bbx):
        self.x = bbx[0]
        self.y = bbx[2]
        self.w = bbx[1] - bbx[0]
        self.h = bbx[3] - bbx[2]


    def bbxScale(self, im_size, scale=1.3):
        assert(scale > 1)
        x = np.around(max(1, self.x - (scale * self.w - self.w) / 2.0))
        y = np.around(max(1, self.y - (scale * self.h - self.h) / 2.0))
        w = np.around(min(scale * self.w, im_size[1] - x))
        h = np.around(min(scale * self.h, im_size[0] - y))
        return BBox([int(x), int(x+w), int(y), int(y+h)])

    def bbxShift(self, im_size, shift=0.03):
        direction = np.random.randn(2)
        x = np.around(max(1, self.x - self.w * shift * direction[0]))
        y = np.around(max(1, self.y - self.h * shift * direction[1]))
        w = min(self.w, im_size[1] - x)
        h = min(self.h, im_size[0] - y)
        return BBox([x, x+w, y, y+h])

    def normalizeLmToBbx(self, landmarks):
        result = []
        lmks = landmarks.copy()
        for lm in lmks:
            lm[0] = (lm[0] - self.x) / self.w
            lm[1] = (lm[1] - self.y) / self.h
            result.append(lm)
        result = np.asarray(result)
        
        return result

