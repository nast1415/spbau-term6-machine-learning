import numpy as np
import cv2


def read_image(path):
    img = np.array(cv2.imread(path))
    height = img.shape[0]
    width = img.shape[1]
    image = np.zeros((height * width, 3))
    for i in range(height):
        for j in range(width):
            image[i * width + j] = img[i][j]
    return image, height, width


def manhattan_dist(x, y):
    return np.sum(np.abs(x - y))


def find_nearest_cluster(clusters, x, distance_metric):
    distance = np.apply_along_axis(distance_metric, 1, clusters, x)
    nearest = distance.argsort()[:1]
    return nearest


def k_means(x, n_clusters, distance_metric):
    centroids = np.random.randint(0, 255, size=(n_clusters, 3))
    print centroids

    nearest_clusters_id = np.zeros(x.shape[0])
    old_centroids = np.zeros((n_clusters, 3))

    iter = 0
    while not np.array_equal(old_centroids, centroids):
        old_centroids = centroids.copy()
        iter += 1
        print iter
        for i in range(x.shape[0]):
            nearest_clusters_id[i] = find_nearest_cluster(centroids, x[i], distance_metric)

        for i in range(n_clusters):
            ith_cluster = np.array(np.where(nearest_clusters_id == i))[0]
            x_slice = x[ith_cluster]
            if x_slice.size == 0:
                arg_val = centroids[i]
            else:
                arg_val = np.mean(x_slice, 0)
            centroids[i] = arg_val

    return nearest_clusters_id, centroids, iter


def centroid_histogram(labels, n_clusters):
    print "Centroid hist!"
    percents = np.zeros(n_clusters)
    size = labels.size
    print n_clusters
    for i in range(n_clusters):
        ith_cluster_labels = np.where(labels == i)[0]
        percents[i] = float(ith_cluster_labels.size) / size
    return percents


def plot_colors(hist, centroids):
    width = 500
    height = 50
    bar = np.zeros((height, width, 3), np.uint8)
    start_x = 0
    for percent, color in zip(hist, centroids):
        end_x = start_x + width * percent
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), height), color.astype("uint8").tolist(), -1)
        start_x = end_x
    return bar


def recolor(image, n_colors):
    img, height, width = read_image(image)
    nearest, centroids, iterations = k_means(img, n_colors, manhattan_dist)

    hist = centroid_histogram(nearest, n_colors)
    bar = plot_colors(hist, centroids)
    cv2.imwrite("bar.png", bar)

    result_image = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            result_image[i][j] = centroids[int(nearest[i * width + j])]
    cv2.imwrite("result.png", result_image)
    return True


def main():
    print recolor(".\images\superman-batman.png", 16)


if __name__ == "__main__":
    main()