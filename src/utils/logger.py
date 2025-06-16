log_lines = []

def get_last_log(n=1):
    return log_lines[-n:] if len(log_lines) >= n else log_lines

def clear_log_lines():
    log_lines.clear()
    
def strip_ansi(text):
    import re
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def log_line(text):
    from sys import stdout
    log_lines.append(strip_ansi(text))
    print(text)  

def flush_logs(filepath):
    global log_lines
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in log_lines:
            f.write(line + '\n')