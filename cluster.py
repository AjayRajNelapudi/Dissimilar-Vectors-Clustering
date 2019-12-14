import numpy as np
from sklearn.cluster import KMeans

class DissimilarVectoredKMeans(KMeans):
    '''
    This class helps in clustering dissimilar vectors of varying length
    '''
    def __init__(self, *args, **kwargs):
        KMeans.__init__(self, *args, **kwargs)

    def build_feature_table(self, X):
        '''
        Builds a list of all unique features in all the input dict vectors

        :param X: input dict vectors
        :return: List of features
        '''
        features = set()
        for datapoint in X:
            for feature in datapoint.keys():
                features.add(feature)
        return list(features)

    def normalize(self, x):
        '''
        Normalizes dissimilar vectors using feature table and saves all values in enlongated vectors

        :param x: dict of features => values
        :return: np array of values whose indices correspond to features in feature table
        '''
        if not hasattr(self, 'features'):
            raise AttributeError('feature table not built')

        total_features = len(self.features)
        vector = np.zeros(shape=(total_features,))
        for attr, value in x.items():
            index = self.features.index(attr)
            vector[index] = value
        return vector

    def build_vectors(self, X):
        '''
        Normalizes all input dict vectors

        :param X: input dict vectors
        :return: np array of normalized vectors
        '''
        vectors = []
        for index in range(len(X)):
            vector = self.normalize(X[index])
            vectors.append(vector)
        return np.array(vectors)

    def denormalize(self, x):
        '''
        Builds dictionary based on feature table and non zero valued indices

        :param x: normalized np array
        :return: dict of features => values
        '''
        if not hasattr(self, 'features'):
            raise AttributeError('feature table not built')

        vector = {}
        for i in range(len(x)):
            if x[i] == 0:
                continue
            feature = self.features[i]
            vector[feature] = x[i]
        return vector

    def build_featured_centers(self):
        '''
        Denormalizes the exisiting cluster_centers_
        :return: list of denormalized cluster centers
        '''
        cluster_centers = []
        for cluster_center in self.cluster_centers_:
            denormalized_center = self.denormalize(cluster_center)
            cluster_centers.append(denormalized_center)
        return cluster_centers

    def fit(self, X, y=None):
        '''
        Trains the model

        :param X: input dict vectors
        :param y: None
        :return: labels
        '''
        self.features = self.build_feature_table(X)
        normalized_X = self.build_vectors(X)
        labels = super().fit(normalized_X)
        self.cluster_centers = self.build_featured_centers()
        return labels

    def predict(self, X):
        '''
        Predicts the output based on trained model
        :param X: input dict vector
        :return: label of predicted center
        '''
        if not hasattr(self, 'features'):
            raise AttributeError('feature table not built')
        if not hasattr(self, 'cluster_centers'):
            raise ValueError('model not trained yet. Consider call predict() after fit()')

        normalized_X = self.build_vectors(X)
        return super().predict(normalized_X)

if __name__ == "__main__":
    training_dataset = [
        {'hello': 2, 'world': 3},
        {'hello': 1, 'world': 2, 'ajay': 3},
        {'world': 3, 'ajay': 2},
        {'hello': 1, 'ajay': 4},
        {'ajay': 5, 'world': 2}
    ]
    model = DissimilarVectoredKMeans(n_clusters=2)
    print('Training Dataset:', training_dataset)
    model.fit(training_dataset)
    print('Labels:', model.labels_)

    print('Cluster centers', model.cluster_centers)

    testing_dataset = [
        {'hello': 1, 'ajay': 1}
    ]
    print('Testing dataset:', testing_dataset)
    label = model.predict(testing_dataset)
    center = model.cluster_centers[label[0]]
    print('Predicted Center:', center)
