from flask import Flask, render_template, request, redirect, url_for
import psycopg2
import smtplib
from datetime import date
from winotify import Notification
from twilio.rest import Client
import pickle
from gensim.models import KeyedVectors
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

classifier = pickle.load(open("trained_model.pkl", "rb"))
encoder = KeyedVectors.load("encoder.kv")

account_sid = "ACd8e4edc61fac1bc1d57e1a57b2affc31"
auth_token = "77b2f848ffd412a6a554361d929a668d"
medapp_no = "+12293675788"

client = Client(account_sid, auth_token)

app = Flask(__name__)
app.secret_key = "medapp1"
logins = {}


def build_connection_with_database():
    conn = psycopg2.connect(database="medapp1", host="localhost", port="5432", user="postgres", password="123")
    return conn


def close_connection_with_database(cur, conn):
    conn.commit()
    cur.close()
    conn.close()


@app.route("/homepage")
def home():
    return render_template("homepage.html")


@app.route('/doctor_login', methods=['POST', 'GET'])
def doctor_login():
    if request.method == "POST":
        name = request.form["name"]
        reg_no = request.form["reg_no"]
        query = f"SELECT * FROM doctors WHERE dr_name = '{name}'"
        conn = build_connection_with_database()
        cur = conn.cursor()
        cur.execute(query)
        data = cur.fetchone()
        reg_no_ = data[1]
        close_connection_with_database(cur, conn)
        if reg_no_ == reg_no:
            logins["doctor"] = name
            return redirect(url_for("doctor_dashboard"))
        else:
            return "Wrong Reg no."
    return render_template("doctor_login.html")


@app.route("/doctor_register", methods=['POST', 'GET'])
def doctor_register():
    if request.method == 'POST':
        name = request.form["name"]
        reg_no = request.form["reg_no"]
        area_code = request.form["area_code"]
        speciality = request.form["specialty"]
        mobile = request.form['mobile']
        hospital = request.form['hospital']
        query = f"INSERT INTO doctors(dr_name, reg_no, area_code, speciality, phone_no, hospital) VALUES ('{name}', '{reg_no}', '{area_code}', '{speciality}', '{mobile}', '{hospital}')"
        table_name = name.split()[0] + name.split()[-1]
        query1 = f"CREATE TABLE {table_name}(patient varchar)"
        query2 = f"CREATE TABLE {table_name}1 (patient varchar, date date)"
        query3 = f"INSERT INTO ACCOUNTS(dr_name, payment) VALUES ('{name}', 0)"
        conn = build_connection_with_database()
        cur = conn.cursor()
        cur.execute(query)
        cur.execute(query1)
        cur.execute(query2)
        cur.execute(query3)
        close_connection_with_database(cur, conn)
        return redirect(url_for("doctor_login"))
    return render_template("doctor_register.html")


@app.route("/user_login", methods=['POST', 'GET'])
def user_login():
    if request.method == 'POST':
        name = request.form["name"]
        password = request.form["password"]
        conn = build_connection_with_database()
        cur = conn.cursor()
        query = f"SELECT * FROM users WHERE name = '{name}'"
        cur.execute(query)
        data = cur.fetchone()
        password_ = data[3]
        close_connection_with_database(cur, conn)
        if password_ == password:
            logins["user"] = name
            return redirect(url_for("user_dashboard_main"))
        else:
            return "Wrong Password!"
    return render_template("user_login.html")


@app.route("/user_register", methods=['POST', 'GET'])
def user_register():
    if request.method == 'POST':
        name = request.form['name']
        area_code = request.form['area_code']
        mobile = request.form['mobile']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        query = f"INSERT INTO users(name, area_code, mobile, password, email, age) VALUES ('{name}', '{area_code}', '{mobile}','{password}', '{email}', '{age}')"
        conn = build_connection_with_database()
        cur = conn.cursor()
        cur.execute(query)
        close_connection_with_database(cur, conn)
        return redirect(url_for("user_login"))
    return render_template("user_register.html")


@app.route('/user_dashboard_main')
def user_dashboard_main():
    return render_template("user_dashboard_main.html")


@app.route("/user_dashboard", methods=['POST', 'GET'])
def user_dashboard():
    conn = build_connection_with_database()
    cur = conn.cursor()
    cur_date = date.today()
    query = f"SELECT * FROM live_dr"
    cur.execute(query)
    data = cur.fetchall()
    close_connection_with_database(cur, conn)
    curr_date = date.today()
    if request.method == 'POST':
        val_ = request.form["book"]
        if val_:
            conn = build_connection_with_database()
            cur = conn.cursor()
            query1 = f"DELETE FROM live_dr WHERE dr_name = '{val_}' "
            table_name = val_.split()[0] + val_.split()[-1]
            query2 = f"INSERT INTO {table_name}(patient) VALUES ('{logins['user']}')"
            query3 = f"INSERT INTO {table_name}1 (patient, date) VALUES ('{logins['user']}', '{curr_date}')"
            query4 = f"INSERT INTO patient(name, dr_name, date) VALUES ('{logins['user']}','{val_}', '{cur_date}')"
            cur.execute(query1)
            cur.execute(query2)
            cur.execute(query3)
            cur.execute(query4)
            close_connection_with_database(cur, conn)
            return redirect(url_for("meeting"))

    #         Twilio se message
    return render_template("user_dashboard.html", data=data)


@app.route("/doctor_dashboard", methods=['POST', 'GET'])
def doctor_dashboard():
    table_name = logins['doctor'].split()[0] + logins['doctor'].split()[-1]
    if request.method == "POST":
        input = request.form["live"]
        if input:
            conn = build_connection_with_database()
            cur = conn.cursor()
            query = f"SELECT * FROM doctors WHERE dr_name = '{logins['doctor']}'"
            cur.execute(query)
            data = cur.fetchone()
            specialty = data[3]
            hospital = data[5]
            query1 = f"INSERT INTO live_dr(dr_name, specialty, hospital) VALUES ('{logins['doctor']}', '{specialty}', '{hospital}')"
            query2 = f"TRUNCATE {table_name}"
            cur.execute(query1)
            cur.execute(query2)
            close_connection_with_database(cur, conn)
            return "You are live"

    conn = build_connection_with_database()
    cur = conn.cursor()
    table_name = logins['doctor'].split()[0] + logins['doctor'].split()[-1]
    query = f"SELECT * FROM {table_name}"
    cur.execute(query)
    data = cur.fetchall()
    if len(data) == 0:
        pass
    elif len(data) == 1:
        notification = Notification(app_id="MEDAPP",
                                    title="Patient's Call",
                                    msg=f"Doctor, Patient is waiting for you.\n")
        notification.show()
    return render_template("doctor_dashboard.html", data=data)


@app.route('/prescription', methods=['POST', 'GET'])
def prescription():
    if request.method == 'POST':
        table_name = logins['doctor'].split()[0] + logins['doctor'].split()[-1]
        conn = build_connection_with_database()
        cur = conn.cursor()
        query = f"SELECT * FROM {table_name}"
        cur.execute(query)
        data = cur.fetchone()
        patient_name = data[0]
        query1 = f"SELECT * FROM users WHERE name = '{patient_name}'"
        cur.execute(query1)
        data1 = cur.fetchone()
        rec_email = data1[4]
        patient_age = data1[5]
        patient_mobile_no = "+91" + data1[2]
        prescription_ = request.form["prescription"]
        # Twilio se SMS for Payment
        message = client.messages.create(body="Kindly Make the payment to get your Prescription,\n"
                                              "payment link : https://buy.stripe.com/test_eVa3eRbd8ey66CQfYY ",
                                         from_=medapp_no,
                                         to=patient_mobile_no)

        # Suppose Payment Received
        message1 = f"""
                Doctor Name : {logins['doctor']} \n
                Patient : {patient_name} \n
                age : {patient_age}\n
                Medicines : {prescription_}
                """
        payment = 250
        query2 = f"UPDATE ACCOUNTS SET payment = {payment} + payment WHERE dr_name = '{logins['doctor']}'"
        cur.execute(query2)
        close_connection_with_database(cur, conn)
        sender_email = "shreekantpukale0@gmail.com"
        password = "idoh gdte pjuo sasb"
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, rec_email, message1)
    return render_template("prescription.html")


@app.route("/patient_history")
def patient_history():
    table_name = logins['doctor'].split()[0] + logins['doctor'].split()[-1]
    conn = build_connection_with_database()
    cur = conn.cursor()
    query = f"SELECT * FROM {table_name}1 ORDER BY date ASC"
    cur.execute(query)
    data = cur.fetchall()
    close_connection_with_database(cur, conn)
    return render_template("patient_history.html", data=data)


@app.route('/patient_history_user')
def patient_history_user():
    conn = build_connection_with_database()
    cur = conn.cursor()
    query = f"SELECT * FROM patient WHERE name = '{logins['user']}' ORDER BY date ASC"
    cur.execute(query)
    data = cur.fetchmany(1)
    dr_mobile_no = 0
    if len(data) >= 1:
        dr_name = data[1]
        query1 = f"SELECT * FROM doctors WHERE dr_name = '{dr_name}'"
        cur.execute(query1)
        data1 = cur.fetchmany(1)
        dr_mobile_no = data1[4]
    close_connection_with_database(cur, conn)
    return render_template("patient_history_user.html", data=data, dr_mobile_no=dr_mobile_no)


@app.route('/logout_doctor')
def logout_doctor():
    return render_template("homepage.html")


@app.route('/meeting')
def meeting():
    return render_template("WEB_UIKITS.html")


# E-Doctor

d2 = pd.read_csv("symptom_Description.csv")
d3 = pd.read_csv("symptom_precaution.csv")
d4 = pd.read_csv("Symptom-severity.csv")
d3.fillna(0, inplace=True)
vec_size = 100


def main(input):
    input = process_text(input)
    vec = np.zeros(vec_size, )
    count = 0
    for word in input:
        if word in encoder.wv.index_to_key:
            vec += encoder.wv[word]
            count += 1
    final_vec = vec / count
    return [final_vec]


stopword = stopwords.words('english')


def process_text(text):
    text = re.sub(r",", "", text)
    stemmer = PorterStemmer()

    clean_text = []
    for word in word_tokenize(text):
        if word not in stopword:
            clean_text.append(stemmer.stem(word))
            # clean_text += " "
    return clean_text


map = {'Fungal infection': 1,
       'Allergy': 2,
       'GERD': 3,
       'Chronic cholestasis': 4,
       'Drug Reaction': 5,
       'Peptic ulcer diseae': 6,
       'AIDS': 7,
       'Diabetes ': 8,
       'Gastroenteritis': 9,
       'Bronchial Asthma': 10,
       'Hypertension ': 11,
       'Migraine': 12,
       'Cervical spondylosis': 13,
       'Paralysis (brain hemorrhage)': 14,
       'Jaundice': 15,
       'Malaria': 16,
       'Chicken pox': 17,
       'Dengue': 18,
       'Typhoid': 19,
       'hepatitis A': 20,
       'Hepatitis B': 21,
       'Hepatitis C': 22,
       'Hepatitis D': 23,
       'Hepatitis E': 24,
       'Alcoholic hepatitis': 25,
       'Tuberculosis': 26,
       'Common Cold': 27,
       'Pneumonia': 28,
       'Dimorphic hemmorhoids(piles)': 29,
       'Heart attack': 30,
       'Varicose veins': 31,
       'Hypothyroidism': 32,
       'Hyperthyroidism': 33,
       'Hypoglycemia': 34,
       'Osteoarthristis': 35,
       'Arthritis': 36,
       '(vertigo) Paroymsal  Positional Vertigo': 37,
       'Acne': 38,
       'Urinary tract infection': 39,
       'Psoriasis': 40,
       'Impetigo': 41}


@app.route('/edoctor', methods=['POST', 'GET'])
def edoctor():
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        # print(symptoms)
        process_symptoms = main(symptoms)
        result = classifier.predict(process_symptoms)
        for k, v in map.items():
            if v == result[0]:
                fr = k
                try:
                    r_description = d2[d2.Disease == fr].Description.values[0]
                except IndexError:
                    pass
                precautions = []
                for i in range(1, 5):
                    p = d3[d3.Disease == fr][f"Precaution_{i}"].values[0]
                    precautions.append(p)
                weight_sum = 0
                message1 = f"You might have {k}"
                message2 = f"Description of {k} is {r_description}"
                message3 = f"Kindly take Following Precautions"
                return render_template("E-Doctor.html", message=message1, message2=message2, message3=message3,pre=precautions)
    return render_template("E-Doctor.html")


if "__main__" == __name__:
    app.run(debug=True)

# Payment
# https://buy.stripe.com/test_eVa3eRbd8ey66CQfYY


#
#

#
#
