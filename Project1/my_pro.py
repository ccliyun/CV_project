from skimage import feature, color, transform, io
import numpy as np
import logging
import matplotlib.pyplot as plt


def compute_edgelets(image, sigma=3):
    gray_img = color.rgb2gray(image)
    edges = feature.canny(gray_img, sigma)
    lines = transform.probabilistic_hough_line(edges, line_length=50,
                                               line_gap=10)

    locations = []
    directions = []
    strengths = []

    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / \
        np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)


def edgelet_lines(edgelets):
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def compute_votes(edgelets, model, threshold_inlier=5):
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    cosine_theta = np.abs(cosine_theta)
    cosine_theta = np.where(cosine_theta>1, 1, cosine_theta)
    theta = np.arccos(cosine_theta)

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths


def ransac_vanishing_point(edgelets, num_ransac_iter=3000, threshold_inlier=10):
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            logging.info("Current best model has {} votes at iteration {}".format(
                current_votes.sum(), ransac_iter))

    return best_model


def remove_inliers(model, edgelets, threshold_inlier=10):
    inliers = compute_votes(edgelets, model, 10) > 0
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets


def vis_edgelets(image, edgelets, show=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    locations, directions, strengths = edgelets
    for i in range(locations.shape[0]):
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]

        plt.plot(xax, yax, 'r-')

    if show:
        plt.show()


def vis_model(image, model, show=True):
    edgelets = compute_edgelets(image)
    locations, directions, strengths = edgelets
    inliers = compute_votes(edgelets, model, 10) > 0

    edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    locations, directions, strengths = edgelets
    vis_edgelets(image, edgelets, False)
    vp = model / model[2]
    plt.plot(vp[0], vp[1], 'bo')
    for i in range(locations.shape[0]):
        xax = [locations[i, 0], vp[0]]
        yax = [locations[i, 1], vp[1]]
        plt.plot(xax, yax, 'b-.')

    if show:
        plt.show()


def cal_height(image, vp1, vp2, t1, t2, b1, b2, v):
    shape = image.shape
    b1 = np.array([b1[0][0], b1[0][1], 1])
    b2 = np.array([b2[0][0], b2[0][1], 1])
    tmp = np.cross(b1, b2)
    vp1 = vp1/vp1[2]
    vp2 = vp2/vp2[2]
    tmp = tmp/tmp[1]
    l = np.cross(vp1, vp2)
    l = l/l[1]
    u = np.cross(tmp,l)
    u = u/u[2]  # cal VP of <b1,b2>

    t1 = np.array([t1[0][0], t1[0][1], 1])
    t2 = np.array([t2[0][0], t2[0][1], 1])
    tmp = np.cross(t1, u)
    tmp = tmp/tmp[1]
    v = v/v[2]
    l2 = np.cross(v,b2)
    l2 = l2/l2[1]
    _t1 =np.cross(tmp,l2)
    _t1 = _t1/_t1[2] # cal intersection of <t1, u> & l2

    # cal distance
    d_t1 = np.linalg.norm(_t1-b2)
    dt2 = np.linalg.norm(t2-b2)
    dv = np.linalg.norm(v-b2)

    # cal scale ratio
    if v[1]<=shape[0]:
        scaled = (d_t1*(dv - dt2))/(dt2*(dv - d_t1))
    else:
        scaled = (d_t1*(dv + dt2))/(dt2*(dv + d_t1))
    vis_cal(image, b1, b2, t1, t2, u, v, _t1, vp1, vp2)
    vis_result(scaled,image,t1, t2, b1, b2)
    return scaled


def vis_cal(image, b1, b2, t1, t2, u, v, _t1, vp1, vp2):
    # vis
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    # plot lines
    xax = [b2[0], b1[0],u[0]]
    yax = [b2[1], b1[1], u[1]]
    plt.plot(xax, yax, 'b-')

    xax = [vp1[0], vp2[0], u[0]]
    yax = [vp1[1], vp2[1], u[1]]
    plt.plot(xax, yax, 'b-.')

    xax = [_t1[0], u[0]]
    yax = [_t1[1], u[1]]
    plt.plot(xax, yax, 'b-')

    xax = [_t1[0], t2[0]]
    yax = [_t1[1], t2[1]]
    plt.plot(xax, yax, 'b-')

    xax = [v[0], b2[0]]
    yax = [v[1], b2[1]]
    plt.plot(xax, yax, 'b-.')
    xax = [v[0], b1[0]]
    yax = [v[1], b1[1]]
    plt.plot(xax, yax, 'b-.')

    xax = [t1[0], b1[0]]
    yax = [t1[1], b1[1]]
    plt.plot(xax, yax, 'r-')

    xax = [t2[0], b2[0]]
    yax = [t2[1], b2[1]]
    plt.plot(xax, yax, 'g-')

    xax = [t1[0], t2[0]]
    yax = [t1[1], t2[1]]
    plt.plot(xax, yax, 'b-')

    # plot dot
    plt.plot(b1[0],b1[1],'ro')
    plt.text(b1[0],b1[1],'b1')

    plt.plot(b2[0], b2[1], 'ro')
    plt.text(b2[0], b2[1], 'b2')

    plt.plot(_t1[0], _t1[1], 'ro')
    plt.text(_t1[0], _t1[1], '~t1')

    plt.plot(u[0], u[1], 'ro')
    plt.text(u[0], u[1], 'u')
    plt.plot(v[0], v[1], 'ro')
    plt.text(v[0], v[1], 'v')

    plt.plot(t1[0], t1[1], 'ro')
    plt.text(t1[0], t1[1], 't1')
    plt.plot(t2[0], t2[1], 'ro')
    plt.text(t2[0], t2[1], 't2')
    plt.show()


def vis_result(scaled, image, t1, t2, b1, b2):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    xax = [t1[0], b1[0]]
    yax = [t1[1], b1[1]]
    plt.plot(xax, yax, 'r-')
    str1="{:.2f}cm".format(scaled*15)
    font = {
            'color': 'red',
            }
    plt.text(t1[0]+5, t1[1]+5,str1,fontdict=font)

    xax = [t2[0], b2[0]]
    yax = [t2[1], b2[1]]
    plt.plot(xax, yax, 'g-')
    font = {
        'color': 'green',
    }
    plt.text(t2[0]+5, t2[1]+5, '15cm', fontdict=font)
    plt.show()


image_name = "calibresult11.jpg"
image = io.imread(image_name)
plt.imshow(image)
print("input reference line base point b2:")
b2 = plt.ginput(1, timeout=0)
print("input reference line top point t2:")
t2 = plt.ginput(1, timeout=0)
print("input unknow line base point b1:")
b1 = plt.ginput(1, timeout=0)
print("input unknow line top point t1:")
t1 = plt.ginput(1, timeout=0)
vp=[]

edgelets1 = compute_edgelets(image)
#vis_edgelets(image, edgelets1, show=False)
vp.append(ransac_vanishing_point(edgelets1, 2000, threshold_inlier=3))
#vis_model(image, vp[0], show=False)
edgelets2 = remove_inliers(vp[0], edgelets1, 10)
#vis_edgelets(image, edgelets2, show=False)
vp.append(ransac_vanishing_point(edgelets2, 2000, threshold_inlier=3))
#vis_model(image, vp[1], False)
edgelets3 = remove_inliers(vp[1], edgelets2, 10)
#vis_edgelets(image, edgelets3, show=False)
vp.append(ransac_vanishing_point(edgelets3, 2000, threshold_inlier=3))
#vis_model(image, vp[2], True)

# find the vertical vanishing point
_b2 = np.array([b2[0][0], b2[0][1]])
_t2 = np.array([t2[0][0], t2[0][1]])
d = []
for i in range(3):
    _vp = vp[i]/vp[i][2]
    _vp = _vp[:2]
    d.append(np.abs(np.cross(_vp-_b2, _t2-_b2))/
             (np.linalg.norm(_vp-_b2)*np.linalg.norm(_t2-_b2)))
d = np.array(d)
print(np.argmin(d))
v = vp[np.argmin(d)]
del(vp[np.argmin(d)])
vp1, vp2 = vp

my_scaled = cal_height(image, vp1, vp2, t1, t2, b1, b2, v)
print(my_scaled*15)
