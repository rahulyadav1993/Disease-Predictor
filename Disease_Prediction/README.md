## ML-Model-Flask-Deployment
This is a Mediclaim Fraud Detection project.

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1 Mediclaim Fraud Detection.py - This contains code for our Machine Learning model to detect fraud on mediclaim data based on training data.
2. app.py - This contains Flask APIs that receives provider details through GUI or API calls, computes the predited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter details and displays the claim is fraud or not.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python Mediclaim Fraud Detection.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000



4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the beow command to send the request with some pre-popuated values -
```
python request.py
```
