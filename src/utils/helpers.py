def get_feature_names():
    return [
        'Total Logon Events',
        'Total File Events',
        'Total Device Events',
        'Total HTTP Events',
        'Logon Count',
        'After Hours Logon',
        'Weekend Logon',
        'File Open Count',
        'File Write Count',
        'File Copy Count',
        'File Delete Count',
        'Device Connect Count',
        'Device Disconnect Count',
        'Visit Frequency',
        'Unique Visit Days',
        'After Hours Browsing',
        'Cloud Service Visits',
        'Job Site Visits'
    ]

def get_risk_classification(prob):
    if prob > 0.7:
        return "Resiko Tinggi"
    elif prob > 0.4:
        return "Resiko Sedang"
    else:
        return "Resiko Rendah (Top Candidate)"

def get_recommendation(prob):
    status = get_risk_classification(prob)
    if status == "Resiko Tinggi":
        return "Segera lakukan investigasi mendalam dan monitoring aktif."
    elif status == "Resiko Sedang":
        return "Perketat pengawasan dan audit aktivitas pengguna."
    else:
        return "Monitor secara rutin dan lakukan edukasi keamanan."

def format_eval_line(label, value, auto_percent=True):
    if auto_percent and isinstance(value, float):
        if 0.0 < value <= 1.0 and "threshold" not in label.lower():
            return f"- {label}: {value * 100:.2f}%"
        else:
            return f"- {label}: {value:.4f}"
    return f"- {label}: {value}"