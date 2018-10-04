import numpy as np
import matplotlib.pyplot as plt


def get_data(file_name):
    inputs = []
    outputs = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.replace('\n','').split(',')
            inputs.append(line[0:2])
            outputs.append(line[2])
    inputs = np.asarray(inputs).astype(np.float64)
    outputs = np.asarray(outputs).astype(np.int16)
    return inputs, outputs


def plot_university_data(features, labels):
    f, ax = plt.subplots(1, 1)
    ax.scatter(features[labels.astype(np.bool), 0], features[labels.astype(np.bool), 1], color="#d00000", label="Admitted")
    ax.scatter(features[~labels.astype(np.bool), 0], features[~labels.astype(np.bool), 1], color="#0000d0", label="Not Admitted")
    ax.set_xlabel("Exam 1 Grade")
    ax.set_ylabel("Exam 2 Grade")
    ax.legend()
    f.set_size_inches(16, 6)
    ax.set_title("University Admission (training set)")
    plt.show()


def add_ones(features):
	return np.hstack((np.ones((features.shape[0],1)), features))


def standardize(features, mean = None, std = None):
	if mean is None or std is None:
		mean = np.mean(features,0)
		std = np.std(features,0)
		return (features - mean) / std, mean, std
	else:
		return (features - mean) / std


def split_data(features, labels, test_ratio, validation_ratio = 0):
    np.random.seed(20)
    shuffle_order = np.random.permutation(features.shape[0])

    features_ = features[shuffle_order]
    labels_ = labels[shuffle_order]
    
    test_size = round(test_ratio * features.shape[0])
    val_size = round(validation_ratio * features.shape[0])
    
    if (val_size == 0):
        train_features = features_[:-test_size]
        train_labels = labels_[:-test_size]
        test_features = features_[-test_size:]
        test_labels = labels_[-test_size:]
        return train_features, train_labels, test_features, test_labels
    else:
        train_features = features_[:-(val_size + test_size)]
        train_labels = labels_[:-(val_size + test_size)]
        val_features = features_[-(val_size + test_size):-test_size]
        val_labels = labels_[-(val_size + test_size):-test_size]
        test_features = features_[-test_size:]
        test_labels = labels_[-test_size:]
        return train_features, train_labels, val_features, val_labels, test_features, test_labels


def plot_university_data_with_line(w, features, labels, title):
    f, ax = plt.subplots(1, 1)
    ax.scatter(features[labels.astype(np.bool), 0], features[labels.astype(np.bool), 1], color="#d00000", label="Admitted")
    ax.scatter(features[~labels.astype(np.bool), 0], features[~labels.astype(np.bool), 1], color="#0000d0", label="Not Admitted")
    ax.set_xlabel("Exam 1 Grade")
    ax.set_ylabel("Exam 2 Grade")
    ax.legend()
    f.set_size_inches(16, 6)
    ax.set_title(f"University Admission ({title})")
    
    limits = ax.get_xlim()
    x1 = np.arange(limits[0], limits[1], 0.001)
    x2 = (-w[0] - w[1] * x1) / w[2]
    ax.plot(x1, x2, c="#00d000")
    
    plt.show()