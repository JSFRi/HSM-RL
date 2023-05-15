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
import seaborn as sns

#other requirements
import io

#Data imports
Tier1 = pd.read_csv("./tier1.csv")
Tier2 = pd.read_csv("./tier2.csv")
Tier3 = pd.read_csv("./tier3.csv")

#app = Flask(__name__, template_folder='./templates',static_folder='./static')
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

    
#Pandas Page
@app.route('/')
@app.route('/pandas', methods=("POST", "GET"))
def Tier():
    return render_template('pandas.html',
                           PageTitle = "Pandas",
                           table=[Tier3.to_html(classes='data', index = False), 
                                  Tier2.to_html(classes='data', index = False),
                                  Tier1.to_html(classes='data', index = False)], titles= ['Tier3','Tier2','Tier1'])


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
    
    fig=plt.figure(1,figsize=(4,2), dpi=320)
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
