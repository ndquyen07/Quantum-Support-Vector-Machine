import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Optional, Union
from enum import Enum


class DatasetType(Enum):
    """Supported dataset types."""
    IRIS = "iris"
    WINE = "wine"
    BREAST_CANCER = "breast_cancer"
    DIGITS = "digits"
    CUSTOM = "custom"


class DataPreprocessor:
    """Handles data loading and preprocessing for SVQSVM."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names = None
        self.target_names = None
    
    def load_dataset(self, 
                    dataset_type: Union[DatasetType, str] = DatasetType.IRIS,
                    samples_per_class: int = 32,
                    custom_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                    test_size: float = 0.3,
                    binary_classes: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess dataset for binary classification.
        
        Args:
            dataset_type: Type of dataset to load
            samples_per_class: Number of samples per class for training (if applicable)
            custom_data: Tuple of (X, y) for custom datasets
            test_size: Proportion of data to use for testing
            binary_classes: Tuple of two class indices for binary classification
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load raw data
        if isinstance(dataset_type, str):
            dataset_type = DatasetType(dataset_type)
            
        X, y = self._load_raw_data(dataset_type, custom_data)
        
        # Convert to binary classification if needed
        if binary_classes is not None:
            X, y = self._filter_binary_classes(X, y, binary_classes)
        elif dataset_type != DatasetType.CUSTOM:
            # Default binary filtering for standard datasets
            X, y = self._default_binary_filter(X, y, dataset_type)
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(
            X, y, samples_per_class, test_size
        )
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Convert labels to {-1, 1}
        y_train = self._convert_labels_to_binary(y_train)
        y_test = self._convert_labels_to_binary(y_test)
        
        return X_train, X_test, y_train, y_test
    
    def _load_raw_data(self, dataset_type: DatasetType, custom_data: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw data based on dataset type."""
        if dataset_type == DatasetType.CUSTOM:
            if custom_data is None:
                raise ValueError("Custom data must be provided for CUSTOM dataset type")
            X, y = custom_data
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            self.target_names = [f"class_{i}" for i in np.unique(y)]
            return X, y
        
        # Load sklearn datasets
        dataset_loaders = {
            DatasetType.IRIS: datasets.load_iris,
            DatasetType.WINE: datasets.load_wine,
            DatasetType.BREAST_CANCER: datasets.load_breast_cancer,
            DatasetType.DIGITS: datasets.load_digits
        }
        
        if dataset_type not in dataset_loaders:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        dataset = dataset_loaders[dataset_type]()
        self.feature_names = dataset.feature_names
        self.target_names = dataset.target_names
        
        return dataset.data, dataset.target
    
    def _filter_binary_classes(self, X: np.ndarray, y: np.ndarray, binary_classes: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Filter data to include only specified binary classes."""
        class1, class2 = binary_classes
        binary_mask = (y == class1) | (y == class2)
        X_filtered = X[binary_mask]
        y_filtered = y[binary_mask]
        
        # Remap labels to 0 and 1
        y_filtered = np.where(y_filtered == class1, 0, 1)
        
        return X_filtered, y_filtered
    
    def _default_binary_filter(self, X: np.ndarray, y: np.ndarray, dataset_type: DatasetType) -> Tuple[np.ndarray, np.ndarray]:
        """Apply default binary filtering for standard datasets."""
        if dataset_type in [DatasetType.IRIS, DatasetType.WINE]:
            # Use first two classes
            binary_mask = y < 2
            X_filtered = X[binary_mask]
            y_filtered = y[binary_mask]
        elif dataset_type == DatasetType.BREAST_CANCER:
            # Already binary
            X_filtered, y_filtered = X, y
        elif dataset_type == DatasetType.DIGITS:
            # Use digits 0 and 1
            binary_mask = (y == 0) | (y == 1)
            X_filtered = X[binary_mask]
            y_filtered = y[binary_mask]
        else:
            X_filtered, y_filtered = X, y
            
        return X_filtered, y_filtered
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, samples_per_class: int, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        # Shuffle data
        np.random.seed(self.config.random_seed)
        shuffled_indices = np.random.permutation(X.shape[0])
        X, y = X[shuffled_indices], y[shuffled_indices]
        
        unique_classes = np.unique(y)
        
        if samples_per_class is not None and len(unique_classes) == 2:
            # Use specified samples per class for training
            train_indices = []
            for class_idx in unique_classes:
                class_indices = np.where(y == class_idx)[0]
                selected_indices = class_indices[:min(samples_per_class, len(class_indices))]
                train_indices.extend(selected_indices)
            
            train_indices = np.array(train_indices)
            test_indices = np.setdiff1d(np.arange(X.shape[0]), train_indices)
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
        else:
            # Use train_test_split with test_size
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, 
                random_state=self.config.random_seed,
                stratify=y
            )
        
        return X_train, X_test, y_train, y_test
    
    def _convert_labels_to_binary(self, y: np.ndarray) -> np.ndarray:
        """Convert labels to {-1, 1} format."""
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(f"Expected binary classification, got {len(unique_labels)} classes")
        
        return np.where(y == unique_labels[0], -1, 1)
    
    def get_dataset_info(self) -> Dict:
        """Get information about the loaded dataset."""
        return {
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'scaler_type': type(self.scaler).__name__
        }
    
    def load_iris_binary(self, samples_per_class: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Legacy method for backward compatibility.
        Use load_dataset(DatasetType.IRIS) instead.
        
        Args:
            samples_per_class: Number of samples per class for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return self.load_dataset(
            dataset_type=DatasetType.IRIS,
            samples_per_class=samples_per_class
        )