import firebase_admin
from firebase_admin import credentials, firestore
import datetime

# Location and side of the camera
LOCATION = "NIT Main Gate"
TRACK_ACTIVITY = "IN"

def initFirebase():
    """
    Function to initialize the firebase instance using the firebase_sdk.json
    credentials file in the file tree
    """

    cred = credentials.Certificate('../firebase_sdk.json')
    firebase_admin.initialize_app(cred)

#######################################################################################

def postPlate(plate):
    """
    Function to post the plate information to the firebase DB

    @param plate: plate text in string format to be posted
    """

    db = firestore.client()
    dateTime = datetime.datetime.now();
    date = dateTime.date()
    time = dateTime.time()
    db.collection("plates").document(str(date)).collection("plates_data").document(str(time)).set({
        "plate": plate,
        "time" : str(time),
        "activity": TRACK_ACTIVITY,
        "location": LOCATION
    })

    # Confirming that the plate has been posted
    print("Plate Posted")