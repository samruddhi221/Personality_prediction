from flask import Flask, render_template, request
import json
from personalityScore import Big5

app = Flask(__name__)
with open('questionaire.json') as f:
    data = json.load(f)

@app.route('/',methods=['GET','POST'])
def Questionnaire():
    return render_template('index.html',result=data)


@app.route('/results',methods=['GET','POST'])
def Results():
    result_temp = {}
    result = {}
    B = Big5()
    try:
        for d in data:
            for key,val in d.items():
                result_temp[key] = int(request.form[val])
        result = B.handle_personality_test(result_temp)
        return render_template('results.html',result=result)
    except:
        print("Error")
if __name__ == "__main__":
    app.run()