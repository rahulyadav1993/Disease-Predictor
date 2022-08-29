## ML-Model-Flask-Deployment
This is a Disease Prediction project.

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model), Flask Azure conginitive services API installed.

### Project Structure

1. app.py - This contains Flask APIs that receives provider details through GUI or API calls, computes the predited value based on our model and returns it.
2. templates - This folder contains the HTML template to allow user to enter details and displays the claim is fraud or not.

### Running the project


1. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

2. Navigate to URL http://localhost:5000



3. You can also send direct POST requests to Flask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
