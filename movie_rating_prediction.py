from flask import Flask, redirect, url_for, render_template,request
import pickle
import pandas as pd
import json


app = Flask(__name__)

@app.route("/", methods=["POST","GET"])
def home():
    with open('./dataset/actors_list.json') as f:
        actor_list = json.load(f) 
    with open('./dataset//director_list.json') as f:
        director_list = json.load(f)
    with open('./dataset//writer_list.json') as f:
        writer_list = json.load(f)

    encoder_actors = pickle.load(open("./model/encoder_actors.pkl", 'rb'))
    encoder_country = pickle.load(open("./model/encoder_country.pkl", 'rb'))
    encoder_director = pickle.load(open("./model/encoder_director.pkl", 'rb'))
    encoder_lan = pickle.load(open("./model/encoder_lan.pkl", 'rb'))
    encoder_writer = pickle.load(open("./model/encoder_writer.pkl", 'rb'))
    encoder_genre = pickle.load(open("./model/encoder_genre.pkl", 'rb'))
    scaler_duration = pickle.load(open("./model/duration_scaler.pkl", 'rb'))
    model = pickle.load(open("./model/random_forest_movie_rating_model.pkl", 'rb'))

    if request.method == "POST":
        #if request.values['send']=='submit':
            genre1 = request.form['genre1']
            genre2 = request.form['genre2']
            genre3 = request.form['genre3']
            actor1 = request.form['actor1']
            actor2 = request.form['actor2']
            if actor2 == "":
                actor2 = "None"
            actor3 = request.form['actor3']
            if actor3 == "":
                actor3 = "None"
            director = request.form['director']
            writer = request.form['writer']
            duration = request.form['duration']
            lan = request.form['lan']
            country = request.form['country']

            moviedata_dict = {
                "genre" : genre1,
                "genre2" : genre2,
                "genre3" : genre3,
                "duration" : duration,
                "country" : country,
                "language" : lan,
                "director" : director,
                "writer" : writer,
                "actors" : actor1,
                "actors2" : actor2,
                "actors3" : actor3               
            }

            info = "movie info: <br/>"+"genre1:"+genre1+"<br/>"+"genre2:"+genre2+"<br/>"+"genre3:"+genre3+"<br/>"+"duration:"+duration+"<br/>"+"country:"+country+"<br/>"+"language:"+lan+"<br/>"+"director:"+director+"<br/>"+"writer:"+writer+"<br/>"+"actor1:"+actor1+"<br/>"+"actor2:"+actor2+"<br/>"+"actor3:"+actor3+"<br/>"

            df = pd.DataFrame(moviedata_dict,index=[0])

            df['language'] = df.language.str.strip()
            df['director'] = df.director.str.strip()
            df['writer'] = df.writer.str.strip()
            df['actors'] = df.actors.str.strip()
            df['actors2'] = df.actors2.str.strip()
            df['actors3'] = df.actors3.str.strip()
            
            df['language'] = df.language.str.replace('[^a-zA-Z ]', '')
            df['director'] = df.director.str.replace('[^a-zA-Z ]', '')
            df['writer'] = df.writer.str.replace('[^a-zA-Z ]', '')
            df['actors'] = df.actors.str.replace('[^a-zA-Z ]', '')
            df['actors2'] = df.actors2.str.replace('[^a-zA-Z ]', '')
            df['actors3'] = df.actors3.str.replace('[^a-zA-Z ]', '')

            df["genre"] = encoder_genre.transform(df["genre"])
            df["genre2"] = encoder_genre.transform(df["genre2"])
            df["genre3"] = encoder_genre.transform(df["genre3"])
            df = encoder_lan.transform(df)
            df = encoder_country.transform(df)
            df = encoder_director.transform(df)
            df = encoder_writer.transform(df)
            df = encoder_actors.transform(df)
            temp_array = scaler_duration.transform(df)
            df = df.reset_index()
            df = df.drop(columns=['index'])
            df["duration"] = pd.DataFrame(temp_array, columns = ['duration'])
            prediction = model.predict(df)
            
            return render_template('main_page.html',score = prediction,actor_list=actor_list,director_list=director_list,writer_list=writer_list,info=info)
    else:
        return render_template("main_page.html",score = "",actor_list=actor_list,director_list=director_list,writer_list=writer_list,info="")



if __name__=="__main__":
    app.run()
