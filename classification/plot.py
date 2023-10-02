import os
import plotly.graph_objects as go
import plotly.express as px

base_dir = os.getcwd()

def plotgraphs(e,acu,val_acu,loss,val_loss,m,accuracy_score):
    file = open(base_dir+"/classification/templates/temp_graphs.html","r+")

    file.truncate(0)

    file.close()



    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,e)), y=acu, name='training accuracy',line=dict(color='firebrick'),mode='lines'))
    fig.add_trace(go.Scatter(x=list(range(1,e)), y=val_acu, name='val accuracy',line=dict(color='royalblue'),mode='lines'))
    fig.update_layout(title='Accuracy', xaxis_title='Epochs',yaxis_title='accuracy')
    with open(base_dir+'/classification/templates/temp_graphs.html', 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,e)), y=loss, name='training loss',line=dict(color='firebrick'),mode='lines'))
    fig.add_trace(go.Scatter(x=list(range(1,e)), y=val_loss, name='val loss',line=dict(color='royalblue'),mode='lines'))
    fig.update_layout(title='Loss', xaxis_title='epochs',yaxis_title='loss')
    with open(base_dir+'/classification/templates/temp_graphs.html', 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


    fig=px.imshow(m)

    fig.update_layout(title='accuracy = {}'.format(accuracy_score))
    with open(base_dir+'/classification/templates/temp_graphs.html', 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
