import numpy as np

def class_similarity(kernel_matrix, y):
    # mask để bỏ đường chéo
    mask = ~np.eye(kernel_matrix.shape[0], dtype=bool)

    # index của từng class
    unique_classes = np.unique(y)
    class_indices = {cls: np.where(y == cls)[0] for cls in unique_classes}

    # Within-class similarities for each class
    within_similarities = []
    for cls in unique_classes:
        idx = class_indices[cls]
        if len(idx) > 1:
            sub_mask = mask[:len(idx), :len(idx)]
            within = kernel_matrix[np.ix_(idx, idx)][sub_mask].mean()
        else:
            within = 0 
        within_similarities.append(within)

    # Average within-class similarity
    average_within = np.mean(within_similarities)

    # Between-class similarities
    between_similarities = []
    for i, cls1 in enumerate(unique_classes):
        for cls2 in unique_classes[i+1:]:
            idx1 = class_indices[cls1]
            idx2 = class_indices[cls2]
            between = kernel_matrix[np.ix_(idx1, idx2)].mean()
            between_similarities.append(between)

    average_between = np.mean(between_similarities) if between_similarities else 0

    # Separability ratio
    sep_ratio = average_within / average_between if average_between != 0 else np.inf

    return within_similarities, average_within, average_between, sep_ratio


def calculate_accuracy(kernel_train, kernel_val, kernel_test, y_train, y_val, y_test):
    from sklearn.svm import SVC
    
    # Use validation set to select optimal C parameter
    # c_values = np.array([0.001, 0.01, 0.1, 1, 10, 100])
    c_values = np.arange(0.001, 101, 0.5)
    val_accuracies = []
    
    for c in c_values:
        # w * x + b 
        svc = SVC(kernel='precomputed', C=c)
        svc.fit(kernel_train, y_train)
        val_acc = svc.score(kernel_val, y_val)
        val_accuracies.append(val_acc)
    
    # Find the best C parameter
    best_idx = np.argmax(val_accuracies)
    best_c = c_values[best_idx]
    best_val_acc = val_accuracies[best_idx]
    
    # If test data provided, evaluate on test set with best C
    svc_final = SVC(kernel='precomputed', C=best_c)
    svc_final.fit(kernel_train, y_train)
    test_acc = svc_final.score(kernel_test, y_test)

    return best_val_acc, test_acc, best_c


def caculate_accuracy_train(kernel_train, y_train):
    from sklearn.svm import SVC
    
    # Use training set to select optimal C parameter
    c_values = np.arange(0.001, 101, 0.5)
    train_accuracies = []
    
    for c in c_values:
        svc = SVC(kernel='precomputed', C=c)
        svc.fit(kernel_train, y_train)
        train_acc = svc.score(kernel_train, y_train)
        train_accuracies.append(train_acc)
    
    # Find the best C parameter
    best_idx = np.argmax(train_accuracies)
    best_train_acc = train_accuracies[best_idx]

    return best_train_acc


