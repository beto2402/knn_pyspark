import math


def knn_classifier(x_train, y_train, element, k):
    distances = []
    dataCollect = x_train.collect()

    # distancia = ((a1_1-a1_2)^2 + (a2_1-a2_2)^2 + ..an)^(1/2)
    for x in dataCollect:
        total_sum = 0
        for i, feature in enumerate(element):
            total_sum += math.pow(feature - x[i], 2)

        distance = math.sqrt(total_sum)

        distances.append(distance)

    res = sorted(range(len(distances)), key=lambda sub: distances[sub])[:k]

    nearest_n = {}

    for index in res:
        target_class = y_train.collect()[index][0]
        
        if target_class not in nearest_n:
            nearest_n[target_class] = 1
        else:
            nearest_n[target_class] += 1

    sorted_nearest = sorted(nearest_n.items(), key=lambda x:x[1], reverse=True)

    return sorted_nearest[0][0]