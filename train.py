import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

vectorizer = DictVectorizer(sparse=False)
le = LabelEncoder()

def extract_features(token):
    #prev_token = ""
    is_number = False
    try:
        if float(token):
            is_number = True
    except:
        pass
    features_dict = {"token": token
        , "lower_cased_token": token.lower()
        #, "prev_token": prev_token
        , "suffix1": token[-1]
        , "suffix2": token[-2:]
        , "suffix3": token[-3:]
        , "is_capitalized": token.upper() == token
        , "is_number": is_number}
    return features_dict


filename = "train.tsv"
f = csv.reader(open(filename), delimiter="\t")
features = []
target = []

for row in f:
  if row:
    feature = extract_features(row[0])
    #print(feature)
    features.append(feature)
    target.append(row[1])

data = vectorizer.fit_transform(features)
label = le.fit(target)
label = label.transform(target)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    data, label, test_size=0.33, random_state=42)
from sklearn.neural_network import MLPClassifier as mlp

model = mlp()
model.fit(x_train, y_train)
t = vectorizer.transform(extract_features("harap"))
le.classes_[model.predict(t)] == "VB"

print(model.score(x_test, y_test))

kalimat = "Jakarta adalah ibukota Indonesa"
for token in kalimat.split():
  vektor = vectorizer.transform(extract_features(token))
  print(token, le.classes_[model.predict(vektor)]) 


