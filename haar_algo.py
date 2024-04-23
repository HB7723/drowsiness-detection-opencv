import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def integral_image(image):
    return np.cumsum(np.cumsum(image, axis=0), axis=1)


def create_features(image):
    return [np.sum(image)]


def prepare_data(images):
    data = []
    for image in images:
        img_integral = integral_image(image)
        features = create_features(img_integral)
        data.extend(features)
    return np.array(data).reshape(-1, 1)


def train_classifier(data, labels):
    model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=50)
    model.fit(data, labels)
    return model


def save_model_to_xml(classifier, filename):
    root = ET.Element("HaarClassifier")
    for stage_id, stage in enumerate(classifier.estimators_):
        stage_elem = ET.SubElement(root, "Stage")
        stage_elem.set('id', str(stage_id))
        for tree_id, tree in enumerate(stage):
            tree_elem = ET.SubElement(stage_elem, "Tree")
            tree_elem.set('id', str(tree_id))
            threshold = tree.tree_.threshold[0]
            left = tree.tree_.value[tree.tree_.children_left[0]]
            right = tree.tree_.value[tree.tree_.children_right[0]]
            threshold_elem = ET.SubElement(tree_elem, "Threshold")
            threshold_elem.text = str(threshold)
            left_elem = ET.SubElement(tree_elem, "Left")
            left_elem.text = str(left[0][0])
            right_elem = ET.SubElement(tree_elem, "Right")
            right_elem.text = str(right[0][0])
    tree = ET.ElementTree(root)
    tree.write(filename)


# Main Execution
train_images = load_images_from_folder('data/train')
train_data = prepare_data(train_images)
train_labels = np.array([1, 0] * (len(train_data) // 2))

model = train_classifier(train_data, train_labels)
save_model_to_xml(model, 'haar_classifier.xml')
