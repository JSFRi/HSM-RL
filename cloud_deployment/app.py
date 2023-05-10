from workerA import add_nums, get_accuracy, get_predictions

from flask import (
   Flask,
   request,
   jsonify,
   Markup,
   render_template,
   send_file,
   make_response,
   url_for,
   Response   
)

#Pandas and Matplotlib
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#other requirements
import io

#Data imports
Tier1 = pd.read_csv("/home/ubuntu/HSM-RL/production_server/tier1.csv")
Tier2 = pd.read_csv("/home/ubuntu/HSM-RL/production_server/tier2.csv")
Tier3 = pd.read_csv("/home/ubuntu/HSM-RL/production_server/tier3.csv")

#app = Flask(__name__, template_folder='./templates',static_folder='./static')
app = Flask(__name__)

@app.route("/")
def index():
    return '<h1>Welcome to the Machine Learning Course.</h1>'

@app.route("/accuracy", methods=['POST', 'GET'])
def accuracy():
    if request.method == 'POST':
        r = get_accuracy.delay()
        a = r.get()
        return '<h1>The accuracy is {}</h1>'.format(a)

    return '''<form method="POST">
    <input type="submit">
    </form>'''

@app.route("/predictions", methods=['POST', 'GET'])
def predictions():
    if request.method == 'POST':
        results = get_predictions.delay()
        predictions = results.get()

        results = get_accuracy.delay()
        accuracy = results.get()
        
        final_results = predictions

        return render_template('result.html', accuracy=accuracy ,final_results=final_results) 
                    
    return '''<form method="POST">
    <input type="submit">
    </form>'''

    
#Pandas Page
@app.route('/')
@app.route('/pandas', methods=("POST", "GET"))
def Tier3():
    return render_template('pandas.html',
                           PageTitle = "Pandas",
                           table=[Tier3.to_html(classes='data', index = False)], titles= Tier3.columns.values)


#Matplotlib page
@app.route('/matplot', methods=("POST", "GET"))
def mpl():
    return render_template('matplot.html',
                           PageTitle = "Matplotlib")


@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    ## Draw heatmap for each tier
    heat1=list(Tier1['temp'])+[0]*(1024-len(Tier1['temp']))
    matrix_heat1=[]
    for i in range(len(heat1)//32):
        matrix_heat1.append(heat1[32*i:(32*i+32)])
    
    heat2=list(Tier2['temp'])+[0]*(1024-len(Tier2['temp']))
    matrix_heat2=[]
    for i in range(len(heat2)//32):
        matrix_heat2.append(heat2[32*i:(32*i+32)])
        
    heat3=list(Tier3['temp'])+[0]*(1024-len(Tier3['temp']))
    matrix_heat3=[]
    for i in range(len(heat3)//32):
        matrix_heat3.append(heat3[32*i:(32*i+32)])
    
    fig=plt.figure(turn,figsize=(12,4), dpi=320)
    plt.subplot(1,3,1)
    plt.title('Tier1')
    sns.heatmap(matrix_heat1,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="Reds")
    plt.subplot(1,3,2)
    plt.title('Tier2')
    sns.heatmap(matrix_heat2,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="Reds")
    plt.subplot(1,3,3)
    plt.title('Tier3')
    sns.heatmap(matrix_heat3,xticklabels=False, yticklabels=False,vmin=0,vmax=1,cmap="Reds")

    return fig


if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5100,debug=True)
