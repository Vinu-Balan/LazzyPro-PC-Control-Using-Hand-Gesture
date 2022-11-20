import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

def train_model():
    df = pd.read_csv('gestures_landmarks_new.csv')
    df.drop(['0_x', '0_y'], axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(df.drop('class_id', axis=1), df['class_id'],
                                                                        test_size=0.30, random_state=1)
    from sklearn.ensemble import RandomForestClassifier
    # Create a Random forest Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, prediction)
    # logmodel = LogisticRegression()
    # logmodel.fit(x_train, y_train)
    # prediction = logmodel.predict(x_test)
    # accuracy = accuracy_score(y_test, prediction)
    print(accuracy)

    filename = 'new_gesture_model.sav'
    pickle.dump(clf, open(filename, 'wb'))