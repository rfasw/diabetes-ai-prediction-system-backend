from flask import Flask, jsonify, request, make_response
import pandas as pd
from joblib import load
from datetime import datetime
from flask_cors import CORS
from weasyprint import HTML
import base64
import os

app = Flask(__name__)
CORS(app)
model = load('model/logregression_model.pkl')
scaler = load('model/scaler.pkl')

# Recommendations for both cases
RECOMMENDATIONS = {
    "Diabetic": [
        "Schedule an appointment with an endocrinologist as soon as possible",
        "Begin a low-glycemic diet focusing on whole foods and vegetables",
        "Start monitoring blood sugar levels daily before meals and at bedtime",
        "Begin moderate exercise regimen (30 mins/day, 5 days/week)",
        "Consider medication management with Metformin as first-line therapy",
        "Schedule quarterly HbA1c tests to monitor long-term glucose control",
        "Attend diabetes education classes for self-management training"
    ],
    "Non-Diabetic": [
        "Continue annual preventive health check-ups",
        "Maintain balanced diet with controlled carbohydrate intake",
        "Engage in regular physical activity (150 mins/week minimum)",
        "Monitor weight and maintain BMI between 18.5-24.9",
        "Limit processed foods and sugary beverages",
        "Get fasting blood glucose test annually",
        "Practice stress-reduction techniques like meditation"
    ]
}

def get_encoded_logo():
    """Try to load and encode logo image from different possible locations"""
    possible_paths = [
        os.path.join('assets', 'diabetes-icon1.png'),  # Relative path 1
        os.path.join('backend', 'assets', 'diabetes-icon1.png'),  # Relative path 2
        os.path.join(os.path.dirname(__file__), 'assets', 'diabetes-icon1.png'),  # Relative to current file
    ]
    
    for logo_path in possible_paths:
        try:
            if os.path.exists(logo_path):
                with open(logo_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error loading logo from {logo_path}: {str(e)}")
            continue
    
    print("Warning: Logo image not found, proceeding without it")
    return None

def generate_pdf_report(patient_data, prediction):
    # Get current date/time
    date_ = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Determine status and recommendations
    status = prediction["status"]
    recommendations = RECOMMENDATIONS[status]
    
    # Try to get encoded logo
    encoded_logo = get_encoded_logo()
    logo_html = f'<img class="logo" src="data:image/png;base64,{encoded_logo}" alt="Clinic Logo">' if encoded_logo else ''
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Diabetes Report - {patient_data['patientId']}</title>
        <style>
            @page {{
                size: A4;
                margin: 0;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f0fff0;
                margin: 0;
                padding: 20px;
                font-size: 14px;
            }}
            .report-card {{
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(46, 125, 50, 0.2);
                width: 100%;
                max-width: 800px;
                padding: 30px;
                margin: 0 auto;
                border-top: 6px solid #2E7D32;
                box-sizing: border-box;
                page-break-after: avoid;
            }}
            .header {{
                text-align: center;
                margin-bottom: 20px;
            }}
            .logo {{
                width: 80px;
                height: 80px;
                margin: 0 auto 10px;
                display: block;
            }}
            .clinic-name {{
                color: #2E7D32;
                font-size: 22px;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .patient-name {{
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                margin: 15px 0;
                color: #2E7D32;
                border-bottom: 2px solid #e0f2e0;
                padding-bottom: 10px;
            }}
            .section {{
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 1px solid #e0f2e0;
            }}
            .section-title {{
                color: #2E7D32;
                font-size: 18px;
                margin-bottom: 12px;
                padding-bottom: 8px;
                border-bottom: 2px solid #e0f2e0;
            }}
            .patient-info {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
            }}
            .info-item {{
                margin-bottom: 8px;
            }}
            .info-label {{
                font-weight: bold;
                color: #388E3C;
                display: block;
                margin-bottom: 3px;
                font-size: 13px;
            }}
            .status-badge {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: bold;
                background-color: {'#ffcdd2' if status == 'Diabetic' else '#c8e6c9'};
                color: {'#b71c1c' if status == 'Diabetic' else '#1b5e20'};
                font-size: 16px;
                margin: 5px 0;
            }}
            .recommendation-list {{
                padding-left: 18px;
                margin: 10px 0;
            }}
            .recommendation-list li {{
                margin-bottom: 8px;
                line-height: 1.5;
            }}
            .footer {{
                text-align: center;
                margin-top: 15px;
                color: #757575;
                font-size: 12px;
            }}
            .watermark {{
                position: absolute;
                opacity: 0.1;
                font-size: 100px;
                font-weight: bold;
                color: #2E7D32;
                transform: rotate(-30deg);
                top: 40%;
                left: 10%;
                z-index: -1;
            }}
            .diagnostic-results {{
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="report-card">
            <div class="header">
                {logo_html}
                <div class="clinic-name">Diabetes AI Clinic</div>
                <div>Advanced Predictive Healthcare Solutions</div>
            </div>
            
            <div class="watermark">DIABETES REPORT</div>
            
            <div class="patient-name">
                {patient_data.get('patientName', 'Patient Name')}
            </div>
            
            <div class="section">
                <h2 class="section-title">Patient Information</h2>
                <div class="patient-info">
                    <div class="info-item">
                        <span class="info-label">Patient ID:</span>
                        {patient_data['patientId']}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Age:</span>
                        {patient_data['age']} years
                    </div>
                    <div class="info-item">
                        <span class="info-label">Blood Sugar:</span>
                        {patient_data['bloodsugar_capped']} mg/dL
                    </div>
                    <div class="info-item">
                        <span class="info-label">Blood Pressure:</span>
                        {patient_data['systolicBp_capped']}/{patient_data['diastolicBp_capped']} mmHg
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Diagnostic Results</h2>
                <div class="diagnostic-results">
                    <div>Prediction Status:</div>
                    <span class="status-badge">{status}</span>
                    <div class="info-item">
                        <span class="info-label">Analysis Date:</span>
                        {date_}
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Medical Recommendations</h2>
                <ol class="recommendation-list">
                    {"".join([f"<li>{rec}</li>" for rec in recommendations])}
                </ol>
            </div>
            
            <div class="footer">
                <p>This report was generated by Diabetes AI Clinic's predictive analysis system</p>
                <p>For questions or concerns, please contact: clinic@diabetesai.com</p>
                <p>Report generated on: {date_}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate PDF
    pdf = HTML(string=html_content).write_pdf()
    return pdf

@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "service": "Diabetes Prediction API",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    classes = ['Non-Diabetic', 'Diabetic']

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON received'}), 400

        # Extract and validate input
        patientId = data.get('patientId')
        patientName = data.get('patientName', '')
        age = data.get('age')
        bloodsugar_capped = data.get('bloodsugar_capped')
        systolicBp_capped = data.get('systolicBp_capped')
        diastolicBp_capped = data.get('diastolicBp_capped')

        required_fields = [patientId, age, bloodsugar_capped, systolicBp_capped, diastolicBp_capped]
        if any(field is None for field in required_fields):
            return jsonify({'error': 'All fields are required'}), 400

        # Validate ranges
        if not (0 < int(age) <= 120):
            return jsonify({'error': 'Age must be between 1 and 120'}), 400
        if not (20 <= float(bloodsugar_capped) <= 1000):
            return jsonify({'error': 'Blood sugar must be between 20 and 1000'}), 400
        if not (50 <= float(systolicBp_capped) <= 250):
            return jsonify({'error': 'Systolic BP must be between 50 and 250'}), 400
        if not (30 <= float(diastolicBp_capped) <= 150):
            return jsonify({'error': 'Diastolic BP must be between 30 and 150'}), 400

        # Prepare input
        new_input = pd.DataFrame([{
            'age': int(age),
            'bloodsugar_capped': float(bloodsugar_capped),
            'systolicBp_capped': float(systolicBp_capped),
            'diastolicBp_capped': float(diastolicBp_capped)
        }])

        # Scale features
        columns_to_scale = ['bloodsugar_capped', 'systolicBp_capped', 'diastolicBp_capped']
        new_input[columns_to_scale] = scaler.transform(new_input[columns_to_scale])

        # Predict
        predicted_class = model.predict(new_input)[0]
        prediction_proba = model.predict_proba(new_input)[0][1]  # Probability of being diabetic
        
        # Get status and recommendations
        status = classes[predicted_class]
        recommendations = RECOMMENDATIONS[status]
        
        return jsonify({
            "patientId": patientId,
            "patientName": patientName,
            "status": status,
            "probability": f"{prediction_proba:.4f}",
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        
        # Generate PDF
        pdf = generate_pdf_report(
            patient_data={
                'patientId': data['patientId'],
                'patientName': data.get('patientName', ''),
                'age': data['age'],
                'bloodsugar_capped': data['bloodsugar_capped'],
                'systolicBp_capped': data['systolicBp_capped'],
                'diastolicBp_capped': data['diastolicBp_capped']
            },
            prediction={
                'status': data['status'],
                'probability': data['probability']
            }
        )
        
        # Create response
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = \
            f'attachment; filename=diabetes_report_{data["patientId"]}.pdf'
            
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)