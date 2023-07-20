import cv2
import numpy as np

def drowLine(image, lines):
    im = image.copy()
    for i in range(len(lines)):
        l = lines[i][0]
        cv2.line(im, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 2, cv2.LINE_AA)
    return im

def display(name, image, size=800):
    height, width, channels = image.shape
    k = size / np.sqrt(width ** 2 + height ** 2)
    w = width * k
    h = height * k
    resizeImage = cv2.resize(image, (int(w), int(h)))
    # cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name, resizeImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getABC(x1, y1, x2, y2):
    '''
    x1, y1: 端点1
    x2, y2: 端点2
    return: 端点1和2构成直线的一般方程系数
    '''
    A = y2 - y1
    B = x1 - x2
    C = (x2 - x1) * y1 - (y2 - y1) * x1
    return A, B, C

def calc_P2P_Distance(x1, y1, x2, y2):
    '''
    x1, y1: 点1
    x2, y2: 点2
    return: 点1和点2的欧式距离
    '''
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

def calc_P2L_Distance(x, y, x1, y1, x2, y2):
    '''
    x, y: 点
    x1, y1, x2, y2: 一条直线的两端端点
    return: 点到该直线的距离
    '''
    A, B, C = getABC(x1, y1, x2, y2)
    denominator = np.sqrt((A ** 2) + (B ** 2))
    if denominator == 0:
        return calc_P2P_Distance(x, y, x1, y1)
    else:
        return abs((A * x) + (B * y) + C) / denominator

def calc_L2L_Distance(xi1, yi1, xi2, yi2, xj1, yj1, xj2, yj2):
    '''
    xi1, yi1, xi2, yi2: 直线1的两端端点
    xi2, yi2, xj2, yj2: 直线2的两端端点
    return: 直线1和直线2的距离
    method: 计算两对端点之间的四种组合的距离，求平均
    '''
    d1 = calc_P2L_Distance(xi1, yi1, xj1, yj1, xj2, yj2)
    d2 = calc_P2L_Distance(xi2, yi2, xj1, yj1, xj2, yj2)
    d3 = calc_P2L_Distance(xj1, yj1, xi1, yi1, xi2, yi2)
    d4 = calc_P2L_Distance(xj2, yj2, xi1, yi1, xi2, yi2)
    return (d1 + d2 + d3 + d4) / 4

def calcK(x1, y1, x2, y2):
    '''
    xi1, yi1, xi2, yi2: 直线的两端端点
    return: 直线的斜率（以角度形式返回）
    '''
    if x2 == x1: return 90
    return np.degrees(np.arctan((y2 - y1) / (x2 - x1)))


def clearClutter(lines, DThreshold = 30, CThreshold = 7):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray）
    DThreshold: 如果两条直线中某对端点距离小于DThreshold，则认为靠近
    CThreshold: 对于一条直线，如果和其靠近的直线数目大于CThreshold，则认定属于杂乱直线
    return: 清理杂乱直线后的结果
    '''
    count = np.ones(lines.shape[0])
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            xi1 = lines[i, 0, 0]
            yi1 = lines[i, 0, 1]
            xi2 = lines[i, 0, 2]
            yi2 = lines[i, 0, 3]
            xj1 = lines[j, 0, 0]
            yj1 = lines[j, 0, 1]
            xj2 = lines[j, 0, 2]
            yj2 = lines[j, 0, 3]
            d1 = calc_P2P_Distance(xi1, yi1, xj1, yj1)
            d2 = calc_P2P_Distance(xi2, yi2, xj1, yj1)
            d3 = calc_P2P_Distance(xi1, yi1, xj2, yj2)
            d4 = calc_P2P_Distance(xi2, yi2, xj2, yj2)
            if min(d1, d2, d3, d4) < DThreshold:
                count[i] += 1
                count[j] += 1
    index = count <= CThreshold
    return lines[index]

def isKNear(line1, line2, maxK=3):
    '''
    line1: 直线1（数据结构：[[x1, y1, x2, y2]], list）
    line2: 直线2（数据结构：[[x1, y1, x2, y2]], list）
    maxK: 在以横轴为x轴和以纵轴为x轴的两种模式下，任意一种斜率的绝对差值小于maxK，则认定直线平行
    return: 直线1和直线2是否平行
    '''
    kx1 = calcK(line1[0][0], line1[0][1], line1[0][2], line1[0][3])
    kx2 = calcK(line2[0][0], line2[0][1], line2[0][2], line2[0][3])
    ky1 = calcK(line1[0][1], line1[0][0], line1[0][3], line1[0][2])
    ky2 = calcK(line2[0][1], line2[0][0], line2[0][3], line2[0][2])
    if abs(kx1 - kx2) < maxK or abs(ky1 - ky2) < maxK:
        return True
    return False

def clearOutside(lines, imShape, threshold, kthreshold=3, maxL2L=20, minL2L=5, maxP2P=20):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray）
    imShape: 图片尺寸
    threshold: 判断是否为外部直线的阈值
    kthreshold: 判断两直线是否平行的阈值
    maxL2L: 平行直线之间的最大距离
    minL2L: 平行直线之间的最小距离
    maxP2P: 平行直线之间对应点的最小距离
    return: 清理掉外部直线之后的直线集合
    method: 1. 如果一条直线的某个端点与图片边界的距离小于threshold，则认定属于外部直线
            2. 对所有的外部直线，如果含有另外一条外部直线和其平行，且直线之间的距离介于[minL2L, maxL2L]之间，
                对应点之间的距离小于maxP2P，则选择其中一条恢复成内部直线
            3. 删除所有外部直线
    '''
    # 步骤1，index保存所有外部直线(对应False, 内部直线对应True)
    x1 = lines[:, :, 0].flatten()
    y1 = lines[:, :, 1].flatten()
    x2 = lines[:, :, 2].flatten()
    y2 = lines[:, :, 3].flatten()
    w = imShape[1]
    h = imShape[0]
    lMax = threshold
    rMax = w - threshold
    tMax = threshold
    bMax = h - threshold
    index = (x1 >= lMax) & (x1 <= rMax) & (y1 >= tMax) & (y1 <= bMax) \
            & (x2 >= lMax) & (x2 <= rMax) & (y2 >= tMax) & (y2 <= bMax)
    # 步骤2
    for i in range(len(index)):
        if index[i] == True: continue # 过滤所有内部直线
        for j in range(len(index)):
            if index[j] == True: continue # 过滤所有内部直线
            if not isKNear(list(lines[j]), list(lines[i])): continue # 过滤非平行直线
            if calc_L2L_Distance(x1[i], y1[i], x2[i], y2[i], x1[j], y1[j], x2[j], y2[j]) > maxL2L: continue
            if calc_L2L_Distance(x1[i], y1[i], x2[i], y2[i], x2[j], y2[j], x1[j], y1[j]) < minL2L: continue
            d1 = calc_P2P_Distance(x1[i], y1[i], x1[j], y1[j])
            d2 = calc_P2P_Distance(x1[i], y1[i], x2[j], y2[j])
            d3 = calc_P2P_Distance(x2[i], y2[i], x1[j], y1[j])
            d4 = calc_P2P_Distance(x2[i], y2[i], x2[j], y2[j])
            if min(min(d1, d2), min(d3, d4)) > maxP2P: continue
            index[i] = True
    # 步骤3
    return lines[index]

def clearShort(lines, threshold):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray)
    threshold: 判断是否短线的阈值
    return: 长度大于阈值的直线
    '''
    x1 = lines[:, :, 0].flatten()
    y1 = lines[:, :, 1].flatten()
    x2 = lines[:, :, 2].flatten()
    y2 = lines[:, :, 3].flatten()
    index = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) > threshold
    return lines[index]

def merge(line1, line2):
    '''
    line1: 直线1（数据结构：[[x1, y1, x2, y2]], list）
    line2: 直线2（数据结构：[[x1, y1, x2, y2]], list）
    return: 直线1和直线2合并后的直线
    method: 选择直线1和直线2中距离最远的两个点，连成一条直线返回
    '''
    x11 = line1[0][0]
    y11 = line1[0][1]
    x12 = line1[0][2]
    y12 = line1[0][3]
    x21 = line2[0][0]
    y21 = line2[0][1]
    x22 = line2[0][2]
    y22 = line2[0][3]
    d1 = calc_P2P_Distance(x11, y11, x21, y21)
    d2 = calc_P2P_Distance(x12, y12, x21, y21)
    d3 = calc_P2P_Distance(x11, y11, x22, y22)
    d4 = calc_P2P_Distance(x12, y12, x22, y22)
    d5 = calc_P2P_Distance(x11, y11, x12, y12)
    d6 = calc_P2P_Distance(x21, y21, x22, y22)
    d_max = max(d1, d2, d3, d4, d5, d6)
    if d_max == d1:
        return [x11, y11, x21, y21]
    elif d_max == d2:
        return [x12, y12, x21, y21]
    elif d_max == d3:
        return [x11, y11, x22, y22]
    elif d_max == d4:
        return [x12, y12, x22, y22]
    elif d_max == d5:
        return [x11, y11, x12, y12]
    else:
        return [x21, y21, x22, y22]

def isNear(line1, line2, flag, kthreshold1=2, kthreshold2=2, maxL2L=5):
    '''
    line1: 直线1（数据结构：[[x1, y1, x2, y2]], list）
    line2: 直线2（数据结构：[[x1, y1, x2, y2]], list）
    flag: 斜率计算标志（为False时以纵轴为x轴，其余情况以横轴为x轴）
    kthreshold1: 如果直线1和直线2的斜率绝对差小于kthreshold1，则直线1和直线2一阶相近
    kthreshold2: 如果直线1和直线2的端点组成的任意一条直线斜率，与其绝对差小于kthreshold2，则直线1和直线2二阶相近
    maxL2L: 如果直线1和直线2的距离小于maxL2L，则直线1和直线2三阶相近
    return: 直线1和直线2在flag指定的模式下是否达到三阶相近
    '''
    # 根据flag设置各个端点
    xi1 = line1[0][0]
    yi1 = line1[0][1]
    xi2 = line1[0][2]
    yi2 = line1[0][3]
    xj1 = line2[0][0]
    yj1 = line2[0][1]
    xj2 = line2[0][2]
    yj2 = line2[0][3]
    if not flag:
        xi1, yi1 = yi1, xi1
        xi2, yi2 = yi2, xi2
        xj1, yj1 = yj1, xj1
        xj2, yj2 = yj2, xj2
    # 是否属于一阶相近
    ki = calcK(xi1, yi1, xi2, yi2)
    kj = calcK(xj1, yj1, xj2, yj2)
    if abs(ki - kj) > kthreshold1: return False
    # 是否属于二阶相近
    k1 = calcK(xi1, yi1, xj1, yj1)
    k2 = calcK(xi1, yi1, xj2, yj2)
    k3 = calcK(xi2, yi2, xj1, yj1)
    k4 = calcK(xi2, yi2, xj2, yj2)
    kmax = max(k1, k2, k3, k4)
    if max(abs(kmax - ki), abs(kmax - kj)) > kthreshold2: return False
    # 是否属于三阶相近
    d = calc_L2L_Distance(xi1, yi1, xi2, yi2, xj1, yj1, xj2, yj2)
    if d > maxL2L: return False
    return True

def isLineNear(line1, line2, kthreshold1=2, kthreshold2=2, maxL2L=5):
    '''
    line1: 直线1（数据结构：[[x1, y1, x2, y2]], list）
    line2: 直线2（数据结构：[[x1, y1, x2, y2]], list）
    return: 直线1和直线2是否相近
    method: 两条直线在横轴为x或纵轴为x的任意一种情况下相近，则返回True；否则，返回False
    '''
    if isNear(line1, line2, True, kthreshold1, kthreshold2, maxL2L) \
            or isNear(line1, line2, False, kthreshold1, kthreshold2, maxL2L):
        return True
    return False

def mergeLines(lines, kthreshold1=2, kthreshold2=2, maxL2L=5):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray）
    return: 合并之后的直线集合
    method: 1. 遍历直线集合，遍历中将相近的直线合并起来覆盖原直线
            2. 如果直线集合改变，重新遍历
    '''
    ts = lines.tolist()
    while True:
        isChange = False
        for i in range(len(ts)):
            if ts[i] == -1: continue
            for j in range(i + 1, len(ts)):
                if ts[j] == -1: continue
                if isLineNear(ts[i], ts[j], kthreshold1, kthreshold2, maxL2L):
                    ts[i] = [merge(ts[i], ts[j])]
                    ts[j] = -1
                    isChange = True
        if not isChange: break
    ts = [x for x in ts if x != -1]
    return np.array(ts)


def extendBlack(imGray):
    '''
    imGray: 灰度图
    return: 黑色像素扩展后的图像
    method: 将黑色像素的四周（八个方向）也变成黑色像素
    '''
    ans = imGray.copy()
    for i in range(1, imGray.shape[0] - 1):
        for j in range(1, imGray.shape[1] - 1):
            if imGray[i, j] == 0:
                ans[i + 1, j] = 0
                ans[i - 1, j] = 0
                ans[i, j + 1] = 0
                ans[i, j - 1] = 0
                ans[i + 1, j + 1] = 0
                ans[i + 1, j - 1] = 0
                ans[i - 1, j + 1] = 0
                ans[i - 1, j - 1] = 0
    return ans

def clearColour(lines, image, maxS=50, minB=10):
    '''
    lines: 直线集合（数据结构：[[[x1, y1, x2, y2]],[[x1, y1, x2, y2]]……], ndarray）
    image: 原始图像
    maxS: 最大饱和度，默认50，落点像素值大于maxS的直线删除
    minB: 最小黑色像素，默认10，落点像素值小于minB的直线删除
    return: 过滤后的直线集合
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channels = hsv[:, :, 1]
    imGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imBlack1 = extendBlack(imGray)
    imBlack2 = extendBlack(imBlack1)
    ts = lines.tolist()
    for j in range(len(ts)):
        if s_channels[ts[j][0][1], ts[j][0][0]] > maxS or s_channels[ts[j][0][3], ts[j][0][2]] > maxS:
            ts[j] = -1
            continue
        if imBlack2[ts[j][0][1], ts[j][0][0]] < minB or imBlack2[ts[j][0][3], ts[j][0][2]] < minB:
            ts[j] = -1
    lines = [x for x in ts if x != -1]
    return np.array(lines)


def detectLane(image):
    imGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imGauss = cv2.GaussianBlur(imGray, (5, 5), 0)
    # Canny边缘检测 + Hough变换
    edges = cv2.Canny(imGauss, 50, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=20, maxLineGap=15)
    lines = clearColour(lines, image) # 清理颜色差异
    lines = clearClutter(lines) # 清理杂线
    lines = clearOutside(lines, image.shape, 20) # 清理外线
    lines = clearShort(lines, 35) # 清理短线
    lines = mergeLines(lines, 2, 2, 5) # 合并直线
    lines = clearShort(lines, 320) # 清理短线
    lines = mergeLines(lines, 5, 5, 10)  # 合并直线
    return lines

def test(i):
    '''
    i: 测试序号
    '''
    if i > 8 or i < 0:
        print('image index out of range')
        exit()
    datadir = "./data"
    name = ['Lab4-01.png', 'Lab4-02.jpg', 'Lab4-03.png', 'Lab4-04.png', 'Lab4-05.png', 'Lab4-06.jpg', 'Lab4-07.jpg', 'Lab4-08.jpg', 'Lab4-09.jpg']
    name = [datadir + '/' + name[i] for i in range(9)]
    # 读取图像
    image = cv2.imread(name[i], cv2.IMREAD_COLOR)
    if image is None:
        print('read image failed')
        exit()
    lines = detectLane(image)
    display('lanes', drowLine(image, lines))

if __name__ == '__main__':
    for i in range(9):
        test(i)