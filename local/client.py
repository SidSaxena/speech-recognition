import requests

# server url
# URL = "http://127.0.0.1/predict"
URL = "http://3.6.86.75/predict"        # aws ec2 instance public address


# audio file we'd like to send for predicting keyword
FILE_PATH = "test/down.wav"


if __name__ == "__main__":

    # open files
    file = open(FILE_PATH, "rb")

    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print("Predicted keyword: {}".format(data["keyword"]))