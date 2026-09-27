import sys
import math

from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

def dist(x, y):
    """
    INPUT: two points x and y
    OUTPUT: the Euclidean distance between two points x and y

    DESCRIPTION: Returns the Euclidean distance between two points.
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))


def parse_line(line):
    """
    INPUT: one line from input file
    OUTPUT: parsed line with numerical values
    
    DESCRIPTION: Parses a line to coordinates.
    """
    values = list(map(float, line.split()))
    return values


def min_dist(point, centroids):
    """
    INPUT: a point and a list of centroids
    OUTPUT: a pair of (minimum distance from point to centroids, closest centroid)

    DESCRIPTION: Returns the minimum distance from a point to a list of centroids.
    """
    closest_centroid = min(centroids, key=lambda centroid: dist(point, centroid))
    return (dist(point, closest_centroid), closest_centroid)


def pick_points(k):
    """
    INPUT: value of k for k-means algorithm
    OUTPUT: the list of initial k centroids.

    DESCRIPTION: Picks the initial cluster centroids for running k-means.
    """
    # - First centroid: the very first data point in the file (deterministic)
    # - Remaining centroids: farthest minimum distance from existing centroids
    path = sys.argv[1]
    points = []
    with open(path, 'r') as f:
        for line in f:
            points.append(parse_line(line))

    centroids = [points[0]]
    for _ in range(1, k):
        next_centroid = max(points, key=lambda point: min_dist(point, centroids)[0])
        centroids.append(next_centroid)
    return centroids


def assign_cluster(centroids, point):
    """
    INPUT: list of centorids and a point
    OUTPUT: a pair of (closest centroid, given point)

    DESCRIPTION: Assigns a point to the closest centroid.
    """
    closest_centroid = min(centroids, key=lambda centroid: dist(point, centroid))
    return (tuple(closest_centroid), point)


def compute_diameter(cluster):
    """
    INPUT: cluster
    OUTPUT: diameter of the given cluster

    DESCRIPTION: Computes the diameter of a cluster.
    """
    # diameter = maximum distance between any two points in the cluster
    if len(cluster) < 2:
        return 0.0
    return max(dist(p1, p2) for i, p1 in enumerate(cluster) for p2 in cluster[i + 1:])


def kmeans(centroids):
    """
    INPUT: list of centroids
    OUTPUT: average diameter of the clusters

    DESCRIPTION: 
    Runs the k-means algorithm and computes the cluster diameters.
    Returns the average diameter of the clusters.

    You may use PySpark things at this function.
    """
    path = sys.argv[1]
    data = sc.textFile(path).map(parse_line)

    # Assign points to clusters
    clustered_data = data.map(lambda point: assign_cluster(centroids, point))
    clusters = clustered_data.groupByKey().mapValues(list)

    # Compute diameters of clusters
    diameters = clusters.mapValues(compute_diameter).values()
    return diameters.mean()


if __name__ == "__main__":
    # ----------------------------------------------
    #           Please do not modify below
    # ----------------------------------------------
    with open('output1.txt', 'w') as f:
        k_value = int(sys.argv[2])
        centroids = pick_points(k_value)
        f.write('k=%d\n' % (k_value))
        f.write('Initial centroids: %s\n' % str(centroids))
        average_diameter = kmeans(centroids)
        f.write('Average diameter: %f\n' % average_diameter)