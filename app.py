from flask import Flask,request,render_template,jsonify
from src.utils import CustomData
from src.pipeline.prediction_pipeline import PredictionPipeline

application=Flask(__name__)
app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_data():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        Data=CustomData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('z')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')
        )

        final_data=Data.get_data_as_dataframe()
        predict_pipeline_obj=PredictionPipeline()
        prediction_value=predict_pipeline_obj.Predict(final_data)

        result=prediction_value[0]

        return render_template('result.html',final_result=result)


if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)